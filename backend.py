from __future__ import annotations
from langchain_core.messages.utils import trim_messages, count_tokens_approximately
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.messages import BaseMessage, SystemMessage
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langgraph.prebuilt import ToolNode, tools_condition
from typing import TypedDict, Annotated, Optional, List
from langgraph.checkpoint.postgres import PostgresSaver
from langchain_community.vectorstores import FAISS
from langchain_core.runnables import RunnableConfig
from langgraph.store.postgres import PostgresStore
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START
import requests, os, tempfile, json, psycopg
from langgraph.store.base import BaseStore
from psycopg_pool import ConnectionPool
from langsmith import traceable, Client
from langgraph.types import interrupt
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import uuid

# =========================== Initialize LLMs and Load environment variables ===========================
load_dotenv()
SHORT_TERM_MEMORY_LIMIT = int(os.getenv("SHORT_TERM_MEMORY_LIMIT", 10))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", 4000))
WEATHER_KEY = os.getenv("WEATHERSTACK_KEY")
ALPHA_KEY = os.getenv("ALPHA_VANTAGE_KEY")
DB_URI = os.getenv("DB_URI")

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
memory_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
client = Client()   


# =========================== Memory Schemas (Pydantic) ===========================
class MemoryItem(BaseModel):
    """
    Schema for a single memory fact.
    """
    text: str = Field(description="Atomic user memory (e.g., 'User likes Python')")
    is_new: bool = Field(description="True if this is a new fact, false if it's already known")

class MemoryDecision(BaseModel):
    """
    Schema for the memory extractor's decision.
    """
    should_write: bool = Field(description="Whether any new memory needs to be written")
    memories: List[MemoryItem] = Field(default_factory=list, description="List of memory items to store")

memory_extractor = memory_llm.with_structured_output(MemoryDecision) # To get the structured output from llm


# =========================== Prompts ===========================
MEMORY_PROMPT = """You are responsible for updating and maintaining accurate user memory.

CURRENT USER DETAILS (existing memories):
{user_details_content}

TASK:
- Review the user's latest message.
- Extract user-specific info worth storing long-term (identity, stable preferences, ongoing projects/goals).
- For each extracted item, set is_new=true ONLY if it adds NEW information compared to CURRENT USER DETAILS.
- If it is basically the same meaning as something already present, set is_new=false.
- Keep each memory as a short atomic sentence.
- No speculation; only facts stated by the user.
- If there is nothing memory-worthy, return should_write=false and an empty list.
"""

SYSTEM_PROMPT_TEMPLATE = """You are a helpful assistant with memory capabilities.
If user-specific memory is available, use it to personalize 
your responses based on what you know about the user.

Your goal is to provide relevant, friendly, and tailored 
assistance that reflects the user’s preferences, context, and past interactions.

If the user’s name or relevant personal context is available, always personalize your responses by:
    – Always Address the user by name (e.g., "Sure, Vivek...") when appropriate
    – Referencing known projects, tools, or preferences (e.g., "your MCP server python based project")
    – Adjusting the tone to feel friendly, natural, and directly aimed at the user

Avoid generic phrasing when personalization is possible.

Use personalization especially in:
    – Greetings and transitions
    – Help or guidance tailored to tools and frameworks the user uses
    – Follow-up messages that continue from past context

Always ensure that personalization is based only on known user details and not assumed.

The user’s memory (which may be empty) is provided as: 
{user_details_content}

{summary_context}

{files_context}

For questions about the uploaded document(s), call the `rag_tool` and include the thread_id `{thread_id}`. If no document is available, ask the user to upload a PDF.
"""


# =========================== PDF retriever store (per thread) ==========================================
STORAGE_DIR = "storage"

# Ensure the main storage directory exists
if not os.path.exists(STORAGE_DIR):
    os.makedirs(STORAGE_DIR)

def _get_thread_storage_path(thread_id: str) -> str:
    """
    Constructs and verifies the directory path for a specific thread's storage.

    Args:
        thread_id (str): The unique identifier for the conversation thread.

    Returns:
        str: The file system path to the thread's storage directory.
    """
    path = os.path.join(STORAGE_DIR, str(thread_id))
    if not os.path.exists(path):
        os.makedirs(path)
    return path

def _load_thread_metadata(thread_id: str) -> dict:
    """
    Loads the metadata JSON file containing information about uploaded files for a thread.

    Args:
        thread_id (str): The unique identifier for the conversation thread.

    Returns:
        dict: A dictionary containing file metadata (e.g., filenames, page counts). 
              Returns {"files": {}} if no metadata exists.
    """
    path = os.path.join(_get_thread_storage_path(thread_id), "metadata.json")
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return {"files": {}}

def _save_thread_metadata(thread_id: str, metadata: dict) -> None:
    """
    Saves the metadata dictionary to a JSON file for a specific thread.

    Args:
        thread_id (str): The unique identifier for the conversation thread.
        metadata (dict): The dictionary containing file information to save.
    """
    path = os.path.join(_get_thread_storage_path(thread_id), "metadata.json")
    with open(path, "w") as f:
        json.dump(metadata, f, indent=2)

def _get_retriever(thread_id: str):
    """
    Loads the FAISS vector index for a specific thread and returns it as a retriever.

    Args:
        thread_id (str): The unique identifier for the conversation thread.

    Returns:
        VectorStoreRetriever | None: A LangChain retriever object if the index exists, 
                                     otherwise None.
    """
    if not thread_id:
        return None
        
    thread_path = _get_thread_storage_path(thread_id)
    index_path = os.path.join(thread_path, "index")
    
    # Check if the FAISS index file actually exists
    if not os.path.exists(os.path.join(index_path, "index.faiss")):
        return None

    try:
        # Load the local FAISS index.
        # allow_dangerous_deserialization=True is needed because FAISS uses pickle.
        vector_store = FAISS.load_local(
            index_path, 
            embeddings, 
            allow_dangerous_deserialization=True
        )
        # Return a retriever configured to find the top 4 similar chunks
        return vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})
    except Exception as e:
        print(f"Error loading vector store: {e}")
        return None

def ingest_pdf(file_bytes: bytes, thread_id: str, filename: str) -> dict:
    """
    Processes a raw PDF file: saves it temporarily, extracts text, creates embeddings,
    updates the FAISS index, and saves metadata.

    Args:
        file_bytes (bytes): The raw binary content of the PDF file.
        thread_id (str): The unique identifier for the conversation thread.
        filename (str): The original name of the uploaded file.

    Returns:
        dict: A summary of the ingestion process, including chunk counts or error messages.
              Example: {"filename": "doc.pdf", "documents": 5, "chunks": 20}
    """
    if not file_bytes:
        raise ValueError("No bytes received for ingestion.")
    
    thread_path = _get_thread_storage_path(thread_id)
    
    # 1. Save the bytes to a temporary file so PyPDFLoader can read it
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(file_bytes)
        temp_path = temp_file.name

    try:
        # 2. Load and Extract Text
        loader = PyPDFLoader(temp_path)
        docs = loader.load()
        
        # Split text into manageable chunks for the LLM context window
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200, separators=["\n\n", "\n", " ", ""]
        )
        chunks = splitter.split_documents(docs)

        if not chunks:
            return {
                "error": "No text found in PDF. It might be a scanned image or empty.",
                "filename": filename,
                "chunks": 0
            }

        # 3. Create or Update FAISS Index
        index_path = os.path.join(thread_path, "index")
        
        if os.path.exists(os.path.join(index_path, "index.faiss")):
            # Load existing index and add new documents
            vector_store = FAISS.load_local(
                index_path, 
                embeddings, 
                allow_dangerous_deserialization=True
            )
            vector_store.add_documents(chunks)
        else:
            # Create a brand new index
            vector_store = FAISS.from_documents(chunks, embeddings)

        # 4. Save the updated index to disk
        vector_store.save_local(index_path)

        # 5. Update Metadata tracking
        metadata = _load_thread_metadata(thread_id)
        metadata["files"][filename] = {
            "chunks": len(chunks),
            "pages": len(docs)
        }
        _save_thread_metadata(thread_id, metadata)

        return {
            "filename": filename,
            "documents": len(docs),
            "chunks": len(chunks)
        }

    except Exception as e:
        return {"error": str(e), "filename": filename, "chunks": 0}
        
    finally:
        # Clean up the temporary file
        try:
            os.remove(temp_path)
        except OSError:
            pass

def get_thread_file_names(thread_id: str) -> List[str]:
    """
    Retrieves a list of filenames currently indexed for a specific thread.

    Args:
        thread_id (str): The unique identifier for the conversation thread.

    Returns:
        List[str]: A list of filenames (e.g., ['report.pdf', 'notes.pdf']).
    """
    if not thread_id:
        return []
    meta = _load_thread_metadata(thread_id)
    return list(meta.get("files", {}).keys())


# =========================== Tools ===========================
@tool
def search(query: str) -> str:
    """
    Search the web using DuckDuckGo.
    
    :param query: Description
    :type query: str
    :return: Description
    :rtype: str
    """
    return DuckDuckGoSearchRun(region="us-en").run(query)

@tool
def calculator(first_num: float, second_num: float, operation: str) -> dict:
    """
    Perform a basic arithmetic operation on two numbers.
    Supported operations: add, sub, mul, div.

    Args:
        first_num (float): The first number.
        second_num (float): The second number.
        operation (str): The operation to perform ('add', 'sub', 'mul', 'div').

    Returns:
        dict: A dictionary containing input arguments and the result, or an error message.
    """
    try:
        if operation == "add":
            result = first_num + second_num
        elif operation == "sub":
            result = first_num - second_num
        elif operation == "mul":
            result = first_num * second_num
        elif operation == "div":
            if second_num == 0:
                return {"error": "Division by zero is not allowed"}
            result = first_num / second_num
        else:
            return {"error": f"Unsupported operation '{operation}'"}
        
        return {"first_num": first_num, "second_num": second_num, "operation": operation, "result": result}
    except Exception as e:
        return {"error": str(e)}

@tool
def get_stock_price(symbol: str) -> dict:
    """
    Fetch the latest stock price for a given symbol using Alpha Vantage.

    Args:
        symbol (str): The stock ticker symbol (e.g., 'AAPL', 'MSFT').

    Returns:
        dict: The JSON response from the Alpha Vantage API containing stock data.
    """
    # For API key : https://www.alphavantage.co/support/#api-key
    url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey={ALPHA_KEY}"
    r = requests.get(url)
    return r.json()

@tool
def purchase_stock(symbol: str, quantity: int) -> dict:
    """
    Simulate purchasing a specific quantity of a stock.
    
    IMPORTANT: This tool implements a Human-in-the-Loop (HITL) workflow.
    It pauses execution to request user confirmation before proceeding.

    Args:
        symbol (str): The stock ticker symbol.
        quantity (int): The number of shares to buy.

    Returns:
        dict: A status message indicating success or cancellation after user input.
    """
    # Trigger an interrupt in the LangGraph workflow and The value passed here is displayed to the user in the frontend.
    decision = interrupt(f"Approve buying {quantity} shares of {symbol}?")

    # Based on the value returned from the frontend (via Command(resume=...))
    if isinstance(decision, str) and decision.lower() == "yes":
        return {
            "status": "success",
            "message": f"Purchase order placed for {quantity} shares of {symbol}.",
            "symbol": symbol,
            "quantity": quantity,
        }
    
    else:
        return {
            "status": "cancelled",
            "message": f"Purchase of {quantity} shares of {symbol} was declined by human.",
            "symbol": symbol,
            "quantity": quantity,
        }

@tool
def get_weather_data(city: str) -> dict:
    """
    Fetches the current weather data for a specified city using WeatherStack.

    Args:
        city (str): The name of the city to look up.

    Returns:
        dict: The JSON response containing weather details (temperature, description, etc.).
    """
    url = f'http://api.weatherstack.com/current?access_key={WEATHER_KEY}&query={city}'
    response = requests.get(url)
    return response.json()

@tool
def rag_tool(query: str, thread_id: Optional[str] = None) -> dict:
    """
    Retrieves context from uploaded PDF documents relevant to the user's query.

    Args:
        query (str): The semantic search query.
        thread_id (str, optional): The ID of the current thread to scope the search.

    Returns:
        dict: Contains the original query, a list of context strings, and source filenames.
    """
    t_id = str(thread_id) if thread_id else None
    
    # Get the specific retriever for this thread
    retriever = _get_retriever(t_id)
    if retriever is None:
        return {
            "error": "No document indexed for this chat. Upload a PDF first.",
            "query": query,
        }

    # Perform the retrieval
    result = retriever.invoke(query)
    
    # Extract clean text and unique sources from results
    context = []
    sources = set()
    
    for doc in result:
        context.append(doc.page_content)
        # Assuming PyPDFLoader puts source path in metadata, we strip to filename
        if "source" in doc.metadata:
            sources.add(os.path.basename(doc.metadata["source"]))

    return {
        "query": query,
        "context": context,
        "sources": list(sources)
    }

# Bind tools to the LLM so it knows their schemas
tools = [search, get_stock_price, purchase_stock, calculator, get_weather_data, rag_tool]
llm_with_tools = llm.bind_tools(tools)

# Create a generic ToolNode that executes the tools when called by the graph
tool_node = ToolNode(tools)


# =========================== State ===========================
class ChatState(TypedDict):
    """
    Represents the state of the conversation graph.
    
    Attributes:
        messages (list[BaseMessage]): A list of messages (System, Human, AI, Tool) 
                                      that acts as the conversation history.
        summary (str): A string that gets overwritten by the summarizer.
    """
    messages: Annotated[list[BaseMessage], add_messages]
    summary: str


# =========================== Nodes ===========================
@traceable(tags=["remember"])
def remember_node(state: ChatState, config: RunnableConfig, *, store: BaseStore) -> dict:
    """
    Analyzes the user's last message to extract and store long-term memories.
    """
    user_id = config["configurable"]["user_id"]
    namespace = ("user", user_id, "details")

    # 1. Retrieve existing memories
    items = store.search(namespace)
    existing_memories = "\n".join(it.value.get("data", "") for it in items) if items else "(empty)"

    # 2. Get the latest user message
    last_message = state["messages"][-1]
    if not isinstance(last_message, BaseMessage) or last_message.type != "human":
        # Only analyze human messages for new memory
        return {}

    # 3. Analyze for new memories
    decision : MemoryDecision = memory_extractor.invoke(
        [
            SystemMessage(content=MEMORY_PROMPT.format(user_details_content=existing_memories)),
            {"role": "user", "content": last_message.content},
        ]
    )

    # 4. Store new memories if found
    if decision.should_write:
        for mem in decision.memories:
            if mem.is_new and mem.text.strip():
                # We use uuid to generate unique keys for each memory item
                store.put(namespace, str(uuid.uuid4()), {"data": mem.text.strip()})
    
    return {}

@traceable(tags=["chat"])
def chat_node(state: ChatState, config : RunnableConfig, *, store: BaseStore) -> dict:
    """
    The main chatbot node. It analyzes the conversation state and decides 
    whether to generate a text response or call a tool.

    Args:
        state (ChatState): The current state of the conversation.
        config (dict, optional): Configuration dictionary containing 'thread_id'.

    Returns:
        dict: A dictionary containing the new message to append to the state.
    """
    thread_id = None
    user_id = "default"

    if config:
        thread_id = config.get("configurable", {}).get("thread_id")
        user_id = config.get("configurable", {}).get("user_id", "default")

    # 1. Fetch Long Term Memory
    namespace = ("user", user_id, "details")
    items = store.search(namespace)
    user_details_content = "\n".join(it.value.get("data", "") for it in items) if items else "(empty)"

    # 2. Fetch available files to inform the LLM about RAG capabilities
    uploaded_files = get_thread_file_names(str(thread_id)) if thread_id else []
    files_context = f"Available documents for RAG: {', '.join(uploaded_files)}" if uploaded_files else "No PDF documents uploaded yet."
    
    # 3. Inject the Summary into the System Message
    summary = state.get("summary", "")
    summary_context = f"Summary of past conversation: {summary}" if summary else "No previous summary."

    # 4. Construct the System Message with dynamic context
    system_message = SystemMessage(
        content=SYSTEM_PROMPT_TEMPLATE.format(
            user_details_content=user_details_content,
            summary_context=summary_context,
            files_context=files_context,
            thread_id=thread_id
        )
    )

    # 5. Trim messages to manage token usage
    trimmed_messages = trim_messages(
        state["messages"][-SHORT_TERM_MEMORY_LIMIT:],
        strategy="last",
        token_counter=count_tokens_approximately,
        max_tokens=MAX_TOKENS
    )

    # 3. Prepend system message to history and invoke LLM
    messages = [system_message, *trimmed_messages]
    response = llm_with_tools.invoke(messages, config=config)
    return {"messages": [response]}

@traceable(tags=["summarize_conversation", str(SHORT_TERM_MEMORY_LIMIT)])
def summarize_conversation(state: ChatState) -> dict:
    """
    Summarizes the conversation history and removes old messages from the database.
    
    Deletes the oldest messages if the history exceeds a certain length.
    This helps keep the DATABASE size manageable.
    
    :param state: Description
    :type state: ChatState
    :return: Description
    :rtype: dict
    """
    existing_summary = state.get("summary", "")
    
    # Construct the prompt for the summarization model
    if existing_summary:
        prompt = (
            f"""
        This is summary of the conversation to date: 

        {existing_summary}

        Extend the summary by taking into account the new messages above.
        """
        )
    else:
        prompt = "Create a summary of the above conversation:"
    
    # We send the messages history except the last N messages + the instruction to summarize
    messages_for_summary = state["messages"][:-SHORT_TERM_MEMORY_LIMIT] + [SystemMessage(content=prompt)]
    
    # Only invoke if there is actually something to summarize other than the prompt
    if len(messages_for_summary) > 1:
        response = llm.invoke(messages_for_summary)
        return {"summary": response.content}
    
    return {}

@traceable(tags=["should_summarize", str(SHORT_TERM_MEMORY_LIMIT)])
def should_summarize(state: ChatState) -> str:
    """
    Determines the next step: Tool? Summarize? or End?
    """
    messages = state["messages"]
    
    # If the conversation is getting long (e.g., > N messages), route to summarizer
    if len(messages) > SHORT_TERM_MEMORY_LIMIT:
        return "summarize_node"
    else:
        return "remember_node"

# =========================== Graph ===========================
# Define the Graph structure
graph = StateGraph(ChatState)

# Nodes
graph.add_node("chat_node", chat_node)
graph.add_node("tools", tool_node)
graph.add_node("remember_node", remember_node)
graph.add_node("summarize_conversation", summarize_conversation)

# Edges
graph.add_conditional_edges(START, should_summarize,
    {
        "remember_node": "remember_node",
        "summarize_node": "summarize_conversation"
    }
)
graph.add_edge('summarize_conversation', 'remember_node')
graph.add_edge("remember_node", "chat_node")
graph.add_conditional_edges("chat_node", tools_condition) # tools_condition decides whether to call a tool or END.
graph.add_edge('tools', 'chat_node') # Return to chat after tool execution


# =========================== DB Setup & Compilation ===========================

# 1. Setup PostgresSaver (Checkpointer)
try:
    """
    Run migrations with a dedicated auto-commit connection as PostgresSaver.setup() creates indexes concurrently, which CANNOT 
    run in a transaction block.
    """
    with psycopg.connect(DB_URI, autocommit=True) as setup_conn:
        PostgresSaver(setup_conn).setup()
    
    with PostgresStore.from_conn_string(DB_URI) as store_setup:
        store_setup.setup()
except Exception as e:
    print(f"Warning during DB setup (indexes might already exist): {e}")

# 2. Setup PostgresStore (Long Term Memory)
# We create ONE connection pool to share between Saver and Store
pool = ConnectionPool(conninfo=DB_URI, max_size=20, kwargs={"autocommit": True})

# 3. Initialize Connections
# Initialize the persistent connection pool. We do NOT use 'with ConnectionPool(...) as pool:' here because the pool needs to
# remain open for the lifetime of the application/module.
checkpointer = PostgresSaver(pool)
store = PostgresStore(pool)

# 4. Compile Graph
# Compile the Graph (passes the checkpointer object at the compilation to add it at each steps)
chatbot = graph.compile(checkpointer=checkpointer, store=store)


# =========================== Helper ===========================
def retrieve_all_threads() -> list:
    """
    Retrieves a list of all unique conversation thread IDs from the database.

    Returns:
        list: A list of thread_id strings.
    """
    all_threads = set()
    # Iterate through all checkpoints to find unique thread IDs from the database
    for checkpoint in checkpointer.list(None):
        cfg = checkpoint.config.get("configurable", {})
        tid = cfg.get("thread_id")
        if tid:
            all_threads.add(tid)

    return list(all_threads)

def get_thread_metadata(thread_id: str) -> dict:
    """
    Wrapper to expose the metadata loader to the frontend.

    Args:
        thread_id (str): The unique identifier for the conversation thread.

    Returns:
        dict: Metadata about the thread's files.
    """
    return _load_thread_metadata(str(thread_id))
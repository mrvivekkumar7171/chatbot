"""
Nodes for the Graph
"""
import uuid

from langchain_core.messages import BaseMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.messages.utils import (
    trim_messages,
    count_tokens_approximately,
)
from langgraph.store.base import BaseStore
from langsmith import traceable

from config.settings import (
    SHORT_TERM_MEMORY_LIMIT,
    MAX_TOKENS
)

from graph.state import ChatState
from graph.memory import (
    MemoryDecision,
    MEMORY_PROMPT,
)

# ==================== Prompts ====================
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

def create_graph_nodes(
    *,
    llm,
    llm_with_tools,
    memory_extractor,
    document_repository,
):
    """_summary_

    Args:
        llm (_type_): _description_
        llm_with_tools (_type_): _description_
        memory_extractor (_type_): _description_
        document_repository (_type_): _description_

    Returns:
        _type_: _description_
    """
    @traceable(tags=["remember"])
    def remember_node(state: ChatState, config: RunnableConfig, *, store: BaseStore) -> dict:
        """
        Analyzes the user's last message to extract and store long-term memories.
        """
        user_id = config["configurable"]["user_id"]
        namespace = ("user", user_id, "details")

        # 1. Retrieve existing memories
        items = store.search(("user", user_id, "details"))
        if items:
            existing_memories = "\n".join(it.value.get("data", "") for it in items)
        else:
            existing_memories = "(empty)"

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
        items = store.search(("user", user_id, "details"))
        if items:
            user_details_content = "\n".join(it.value.get("data", "") for it in items)
        else:
            user_details_content = "(empty)"

        # 2. Fetch available files to inform the LLM about RAG capabilities
        if thread_id:
            uploaded_files = document_repository.get_thread_file_names(str(thread_id))
        else:
            uploaded_files = []
        if uploaded_files:
            files_context = f"Available documents for RAG: {', '.join(uploaded_files)}"
        else:
            files_context = "No PDF documents uploaded yet."

        # 3. Inject the Summary into the System Message
        summary = state.get("summary", "")
        if summary:
            summary_context = f"Summary of past conversation: {summary}"
        else:
            summary_context = "No previous summary."

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
        Summarizes the conversation history and reduce old messages from the state.

        Remove the oldest messages if the history exceeds a certain length.
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
        messages_for_summary = state["messages"][:-SHORT_TERM_MEMORY_LIMIT]
        messages_for_summary.append(SystemMessage(content=prompt))

        # Only invoke if there is actually something to summarize other than the prompt
        if len(messages_for_summary) > 1:
            response = llm.invoke(messages_for_summary)
            return {"summary": response.content}

        return {}

    return remember_node, chat_node, summarize_conversation

"""
Backend for the LangGraph AI chatbot, handling LLM calls, memory, tools, PDF processing, and RAG.
Uses Groq for the LLM, Gemini hosted embeddings, and Supabase for storage,
PostgreSQL, pgvector, chat history, checkpoints, and long-term user memory.
"""
from __future__ import annotations

from llm.chat import create_chat_llm
from llm.factory import create_memory_llm
from llm.embeddings import create_embeddings

from db.supabase import create_supabase_client
from db.repositories.document_repository import DocumentRepository
from db.repositories.vector_repository import VectorRepository
from db.repositories.conversation_history_repository import (
    ConversationHistoryRepository,
)
from db.connections import create_connection_pool
from db.langgraph import (
    setup_langgraph_database,
    create_checkpointer,
    create_store,
    get_all_thread_ids,
)

from rag.retrieval import RAGRetriever
from rag.ingestion import RAGIngestion
from rag.service import RAGService

from tools.weather import create_weather_tool
from tools.stocks import create_stock_price_tool
from tools.web_search import create_web_search_tool
from tools_registry import create_tool_registry
from tools.purchase import purchase_stock
from tools.calculator import calculator
from tools.rag import create_rag_tool
from tools.sandbox import safe_write_code, shell_tool

from graph.builder import build_chatbot



# ==================== Initialize LLMs and DataBase Client ====================
llm = create_chat_llm()
memory_llm = create_memory_llm()
embeddings = create_embeddings()
supabase = create_supabase_client()



# ==================== RAG Store ====================
# Retrieves document metadata from Supabase
document_repository = DocumentRepository(supabase)
# Retrieves document chunks from Supabase
vector_repository = VectorRepository(supabase)
conversation_history_repository = ConversationHistoryRepository(supabase)

rag_service = RAGService(
    # upload document, split into chunks, create embeddings, store in Supabase
    ingestion=RAGIngestion(
        supabase_client=supabase,
        embeddings=embeddings
    ),
    # retrieve document chunks from Supabase for a query
    retriever=RAGRetriever(
        embeddings=embeddings,
        vector_repository=vector_repository
    )
)



# ==================== Tools  Registory ====================
search = create_web_search_tool()
get_stock_price = create_stock_price_tool()
get_weather_data = create_weather_tool()
rag_tool = create_rag_tool(rag_service)

# Bind tools to the LLM so it knows their schemas
tools = create_tool_registry(
    search=search,
    get_stock_price=get_stock_price,
    get_weather_data=get_weather_data,
    rag_tool=rag_tool,
    purchase_stock=purchase_stock,
    calculator=calculator,
    shell_tool=shell_tool,
    safe_write_code=safe_write_code,
)



# ==================== DB Setup & Compilation ====================

# 1. Setup PostgresSaver (Checkpointer) and PostgresStore (Long Term Memory)
setup_langgraph_database()

# 2. Initialize ONE ConnectionPool to share between Saver and Store
pool = create_connection_pool()
checkpointer = create_checkpointer(pool)
postgres_store = create_store(pool)

# 3. Build the Chatbot
chatbot = build_chatbot(
    llm=llm,
    tools=tools,
    store=postgres_store,
    checkpointer=checkpointer,
    memory_llm=memory_llm,
    document_repository=document_repository,
    embeddings=embeddings,
    conversation_history_repository=conversation_history_repository,
)



# ==================== Helper ====================

def ingest_pdf(file_bytes: bytes, thread_id: str, filename: str) -> dict:
    """
    Processes a raw PDF file: saves it temporarily, extracts text,
    creates embeddings, and stores them in Supabase.

    Args:
        file_bytes (bytes): The raw binary content of the PDF file.
        thread_id (str): The unique identifier for the conversation thread.
        filename (str): The original name of the uploaded file.

    Returns:
        dict: A summary of the ingestion process, including chunk counts or error messages.
              Example: {"filename": "doc.pdf", "documents": 5, "chunks": 20}
    """
    return rag_service.ingest_pdf(
        file_bytes=file_bytes,
        thread_id=thread_id,
        filename=filename,
    )

def get_thread_metadata(thread_id: str) -> dict:
    """Return uploaded-document metadata for a conversation thread."""
    return document_repository.get_thread_metadata(thread_id)

def retrieve_all_threads() -> list:
    """Return all conversation thread IDs stored by the checkpointer."""
    return get_all_thread_ids(checkpointer)

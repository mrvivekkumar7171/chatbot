"""
Backend for the LangGraph AI chatbot, handling LLM calls, memory, tools, PDF processing, and RAG.
Uses Groq for the LLM, Hugging Face embeddings for vector generation, and Supabase for storage,
PostgreSQL, pgvector, chat history, checkpoints, and long-term user memory.
"""
from __future__ import annotations

from llm.factory import create_chat_llm, create_memory_llm
from llm.embeddings import create_embeddings

from db.connections import create_connection_pool
from db.supabase import create_supabase_client
from db.repositories.document_repository import DocumentRepository
from db.repositories.vector_repository import VectorRepository
from db.langgraph import (
    setup_langgraph_database,
    create_checkpointer,
    create_store,
    get_all_thread_ids,
)

from rag.retrieval import RAGRetriever
from rag.ingestion import RAGIngestion
from rag.service import RAGService

from clients.weather_api import WeatherAPIClient
from clients.web_search import WebSearchClient
from clients.alpha_vantage import AlphaVantageClient

from services.weather_service import WeatherService
from services.stock_service import StockService
from services.search_service import SearchService

from tools.weather import create_weather_tool
from tools.stocks import create_stock_price_tool
from tools.web_search import create_web_search_tool
from tools.registry import create_tool_registry
from tools.purchase import purchase_stock
from tools.calculator import calculator
from tools.rag import create_rag_tool

from graph.memory import create_memory_extractor
from graph.builder import build_chatbot

from config.settings import DATABASE_URL



# ==================== Initialize LLMs and Load environment variables ====================
llm = create_chat_llm()
memory_llm = create_memory_llm()
embeddings = create_embeddings()
supabase = create_supabase_client()



# ==================== PDF retriever store (per thread) ====================
document_repository = DocumentRepository(supabase)
vector_repository = VectorRepository(supabase)
rag_retriever = RAGRetriever(
    embeddings=embeddings,
    vector_repository=vector_repository,
)
rag_ingestion = RAGIngestion(
    supabase_client=supabase,
    embeddings=embeddings,
)
rag_service = RAGService(
    ingestion=rag_ingestion,
    retriever=rag_retriever,
)

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



# ==================== Tools ====================
search_client = WebSearchClient()
search_service = SearchService(search_client)
search = create_web_search_tool(search_service)

stock_client = AlphaVantageClient()
stock_service = StockService(stock_client)
get_stock_price = create_stock_price_tool(stock_service)

weather_client = WeatherAPIClient()
weather_service = WeatherService(weather_client)
get_weather_data = create_weather_tool(weather_service)

rag_tool = create_rag_tool(rag_service)

# Bind tools to the LLM so it knows their schemas
tools = create_tool_registry(
    search=search,
    calculator=calculator,
    get_stock_price=get_stock_price,
    purchase_stock=purchase_stock,
    get_weather_data=get_weather_data,
    rag_tool=rag_tool,
)
memory_extractor = create_memory_extractor(memory_llm)



# ==================== DB Setup & Compilation ====================

# 1. Setup PostgresSaver (Checkpointer)
setup_langgraph_database(DATABASE_URL)

# 2. Setup PostgresStore (Long Term Memory)
# We create ONE connection pool to share between Saver and Store
pool = create_connection_pool()

# 3. Initialize Connections
checkpointer = create_checkpointer(pool)

# 4. Build the Chatbot
chatbot = build_chatbot(
    llm=llm,
    tools=tools,
    checkpointer=checkpointer,
    store=create_store(pool),
    memory_extractor=memory_extractor,
    document_repository=document_repository,
)



# ==================== Helper ====================
def get_thread_file_names(thread_id: str) -> list[str]:
    """_summary_

    Args:
        thread_id (str): _description_

    Returns:
        list[str]: _description_
    """
    return document_repository.get_thread_file_names(thread_id)

def get_thread_metadata(thread_id: str) -> dict:
    """_summary_

    Args:
        thread_id (str): _description_

    Returns:
        dict: _description_
    """
    return document_repository.get_thread_metadata(thread_id)

def retrieve_all_threads() -> list:
    """_summary_

    Returns:
        list: _description_
    """
    return get_all_thread_ids(checkpointer)

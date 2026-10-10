"""
Settings for the application.
"""
import os

from dotenv import load_dotenv

load_dotenv()


def _required(name: str) -> str:
    """Return a required environment variable or raise a clear error."""
    value = os.getenv(name)
    if value is None or not value.strip():
        raise ValueError(f"{name} must be declared in the .env file.")
    return value


def _integer(name: str) -> int:
    """Parse a required positive integer environment variable."""
    value = int(_required(name))
    if value <= 0:
        raise ValueError(f"{name} must be greater than zero.")
    return value


SHORT_TERM_MEMORY_LIMIT = _integer("SHORT_TERM_MEMORY_LIMIT")
LONG_TERM_MEMORY_INTERVAL = _integer("LONG_TERM_MEMORY_INTERVAL")
MAX_TOKENS = _integer("MAX_TOKENS")
LONG_CONTEXT_MATCH_LIMIT = _integer("LONG_CONTEXT_MATCH_LIMIT")
RECENT_HISTORY_TURNS = _integer("RECENT_HISTORY_TURNS")
MAX_ITERATIONS = _integer("MAX_ITERATIONS")
RAG_MAX_PAGES_LIMIT = _integer("RAG_MAX_PAGES_LIMIT")
TEXT_SPLITTER_CHUNK_SIZE = _integer("TEXT_SPLITTER_CHUNK_SIZE")
TEXT_SPLITTER_OVERLAP = _integer("TEXT_SPLITTER_OVERLAP")

SUPABASE_SERVICE_KEY = _required("SUPABASE_SERVICE_KEY")
WEATHER_KEY = _required("WEATHERSTACK_KEY")
ALPHA_KEY = _required("ALPHA_VANTAGE_KEY")
DATABASE_URL = _required("DATABASE_URL")
SUPABASE_URL = _required("SUPABASE_URL")
GOOGLE_API_KEY = _required("GOOGLE_API_KEY")
EMBEDDING_MODEL = _required("EMBEDDING_MODEL")
EMBEDDING_DIMENSION = _integer("EMBEDDING_DIMENSION")

CHAT_MODEL = _required("CHAT_MODEL")
CHAT_TEMPERATURE = float(_required("CHAT_TEMPERATURE"))
if not 0 <= CHAT_TEMPERATURE <= 2:
    raise ValueError("CHAT_TEMPERATURE must be between 0 and 2.")
CURRENT_USER_ID = _required("CURRENT_USER_ID")
LANGGRAPH_RUN_NAME = _required("LANGGRAPH_RUN_NAME")
LANGGRAPH_PARSER = _required("LANGGRAPH_PARSER")
LANGGRAPH_TAGS = _required("LANGGRAPH_TAGS").split(",")
APP_LAYOUT = _required("APP_LAYOUT")
APP_PAGE_TITLE = _required("APP_PAGE_TITLE")
APP_TITLE = _required("APP_TITLE")
APP_PAGE_ICON = _required("APP_PAGE_ICON")
PDF_FILE_EXTENSION = _required("PDF_FILE_EXTENSION")
WEB_SEARCH_REGION = _required("WEB_SEARCH_REGION")
ALPHA_VANTAGE_URL = _required("ALPHA_VANTAGE_URL")
WEATHERSTACK_URL = _required("WEATHERSTACK_URL")
HTTP_TIMEOUT_SECONDS = _integer("HTTP_TIMEOUT_SECONDS")
DATABASE_POOL_MAX_SIZE = _integer("DATABASE_POOL_MAX_SIZE")
AGENT_WORKSPACE = _required("AGENT_WORKSPACE")
SANDBOX_COMMAND_TIMEOUT_SECONDS = _integer("SANDBOX_COMMAND_TIMEOUT_SECONDS")

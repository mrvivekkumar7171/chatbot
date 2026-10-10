"""
LLM Factory for creating language model instances.
"""
from langchain_groq import ChatGroq

from config.settings import CHAT_MODEL, CHAT_TEMPERATURE


def create_memory_llm():
    """Create the Groq model used for durable-memory extraction."""
    return ChatGroq(
        model=CHAT_MODEL,
        temperature=CHAT_TEMPERATURE,
    )

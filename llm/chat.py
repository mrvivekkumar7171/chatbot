"""
LLM Factory for creating language model instances.
"""
from langchain_groq import ChatGroq

from config.settings import CHAT_MODEL, CHAT_TEMPERATURE


def create_chat_llm():
    """Create the configured Groq chat model."""
    return ChatGroq(
        model=CHAT_MODEL,
        temperature=CHAT_TEMPERATURE,
    )

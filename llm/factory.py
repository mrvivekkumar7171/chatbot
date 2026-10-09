"""
LLM Factory for creating language model instances.
"""
from langchain_groq import ChatGroq

def create_memory_llm():
    """_summary_

    Returns:
        _type_: _description_
    """
    return ChatGroq(
        model="openai/gpt-oss-20b",
        temperature=1,
    )

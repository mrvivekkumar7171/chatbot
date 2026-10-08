"""
Routing logic for the LangGraph AI chatbot.
"""
from langsmith import traceable
from config.settings import (
    SHORT_TERM_MEMORY_LIMIT
)
from graph.state import ChatState

@traceable(tags=["should_summarize", str(SHORT_TERM_MEMORY_LIMIT)])
def should_summarize(state: ChatState) -> str:
    """
    Determines the next step: Tool? Summarize? or End?
    """
    messages = state["messages"]

    # If the conversation is getting long (e.g., > N messages), route to summarizer
    if len(messages) > SHORT_TERM_MEMORY_LIMIT:
        return "summarize_node"
    return "remember_node"

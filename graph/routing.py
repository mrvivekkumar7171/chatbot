"""
Routing logic for the LangGraph AI chatbot.
"""
from langsmith import traceable
from graph.state import ChatState
from config.settings import LONG_TERM_MEMORY_INTERVAL
from langchain_core.messages import HumanMessage

@traceable(tags=["is_long_chat", str(LONG_TERM_MEMORY_INTERVAL)])
def is_long_chat(state: ChatState) -> str:
    """
    Determines the next step: chat? or check for Long Term Memory?
    For Every nth Human Message
    """
    messages = state["messages"]

    count = 0
    for message in messages:
        if isinstance(message, HumanMessage):
            count += 1

    if count % LONG_TERM_MEMORY_INTERVAL == 0:
        return "long_chat"
    return "short_chat"

"""
State schemas for the LangGraph AI chatbot.
"""
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


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

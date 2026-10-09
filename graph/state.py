"""
State schemas for the LangGraph AI chatbot.
"""
from typing import TypedDict, Annotated, NotRequired
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class ChatState(TypedDict):
    """
    Represents the state of the conversation graph.
    
    Attributes:
        messages (list[BaseMessage]): A list of messages (System, Human, AI, Tool) 
                                      that acts as the conversation history.
    """
    messages: Annotated[list[BaseMessage], add_messages]
    ltm_processed_human_count: NotRequired[int]

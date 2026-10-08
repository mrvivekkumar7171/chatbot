"""
Memory schemas and prompts for the LangGraph AI chatbot.
"""
from typing import List
from pydantic import BaseModel, Field

# ==================== Prompts ====================
MEMORY_PROMPT = """You are responsible for updating and maintaining accurate user memory.

CURRENT USER DETAILS (existing memories):
{user_details_content}

TASK:
- Review the user's latest message.
- Extract user-specific info worth storing long-term (identity, stable preferences, ongoing projects/goals).
- For each extracted item, set is_new=true ONLY if it adds NEW information compared to CURRENT USER DETAILS.
- If it is basically the same meaning as something already present, set is_new=false.
- Keep each memory as a short atomic sentence.
- No speculation; only facts stated by the user.
- If there is nothing memory-worthy, return should_write=false and an empty list.
"""

# ==================== Memory Schemas (Pydantic) ====================
class MemoryItem(BaseModel):
    """
    Schema for a single memory fact.
    """
    text: str = Field(description="Atomic user memory (e.g., 'User likes Python')")
    is_new: bool = Field(description="True if this is a new fact, false if it's already known")

class MemoryDecision(BaseModel):
    """
    Schema for the memory extractor's decision.
    """
    should_write: bool = Field(description="Whether any new memory needs to be written")
    memories: List[MemoryItem] = Field(
        default_factory=list,
        description="List of memory items to store"
        )

def create_memory_extractor(memory_llm):
    """To get the structured output from llm"""
    return memory_llm.with_structured_output(
        MemoryDecision,
        method="json_schema",
    )

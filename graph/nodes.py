"""
Nodes for the Graph
"""
import uuid

from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage
from langchain_core.runnables import RunnableConfig
from langgraph.store.base import BaseStore
from langsmith import traceable

from config.settings import (
    LONG_CONTEXT_MATCH_LIMIT,
    RECENT_HISTORY_TURNS,
)
from graph.state import ChatState
from graph.memory import (
    create_memory_extractor,
    MemoryDecision,
    MEMORY_PROMPT,
)

# ==================== Prompts ====================
SYSTEM_PROMPT_TEMPLATE = """You are a helpful assistant with memory capabilities.
If user-specific memory is available, use it to personalize 
your responses based on what you know about the user.

Your goal is to provide relevant, friendly, and tailored 
assistance that reflects the user’s preferences, context, and past interactions.

If the user’s name or relevant personal context is available, always personalize your responses by:
    – Always Address the user by name (e.g., "Sure, Vivek...") when appropriate
    – Referencing known projects, tools, or preferences (e.g., "your MCP server python based project")
    – Adjusting the tone to feel friendly, natural, and directly aimed at the user

Avoid generic phrasing when personalization is possible.

Use personalization especially in:
    – Greetings and transitions
    – Help or guidance tailored to tools and frameworks the user uses
    – Follow-up messages that continue from past context

Always ensure that personalization is based only on known user details and not assumed.

The user’s memory (which may be empty) is provided as: 
{user_details_content}

{summary_context}

{files_context}

For questions about the uploaded document(s), call the `rag_tool` and include the thread_id `{thread_id}`. If no document is available, ask the user to upload a PDF.

Before changing files, inspect them with `shell_tool`. Use `safe_write_code`
for all file writes; never use shell redirection, pipes, or destructive commands.
File writes are limited to the dedicated agent workspace.
"""
def create_graph_nodes(
    *,
    llm,
    llm_with_tools,
    memory_llm,
    document_repository,
    embeddings,
    conversation_history_repository,
):
    """_summary_

    Args:
        llm (_type_): _description_
        llm_with_tools (_type_): _description_
        memory_llm (_type_): _description_
        document_repository (_type_): _description_

    Returns:
        _type_: _description_
    """

    memory_llm = create_memory_extractor(memory_llm)

    @traceable(tags=["remember"])
    def long_term_memory(state: ChatState, config: RunnableConfig, *, store: BaseStore) -> dict:
        """
        Analyzes all human messages since the previous memory update.
        """
        configurable = config.get("configurable", {})
        user_id = configurable.get("user_id")
        thread_id = configurable.get("thread_id")
        if not user_id or not thread_id:
            raise ValueError("Both user_id and thread_id are required.")
        namespace = ("user", user_id, "details")

        human_messages = [
            message
            for message in state["messages"]
            if isinstance(message, HumanMessage)
        ]
        processed_count = state.get("ltm_processed_human_count", 0)
        unprocessed_messages = human_messages[processed_count:]

        for message in unprocessed_messages:
            items = store.search(namespace)
            existing_memories = (
                "\n".join(it.value.get("data", "") for it in items)
                if items
                else "(empty)"
            )
            decision: MemoryDecision = memory_llm.invoke(
                [
                    SystemMessage(
                        content=MEMORY_PROMPT.format(
                            user_details_content=existing_memories
                        )
                    ),
                    {"role": "user", "content": message.content},
                ]
            )

            if decision.should_write:
                for memory in decision.memories:
                    if memory.is_new and memory.text.strip():
                        store.put(
                            namespace,
                            str(uuid.uuid4()),
                            {"data": memory.text.strip()},
                        )

        return {"ltm_processed_human_count": len(human_messages)}

    @traceable(tags=["chat"])
    def chat_node(state: ChatState, config: RunnableConfig, *, store: BaseStore) -> dict:
        """
        The main chatbot node. It analyzes the conversation state and decides 
        whether to generate a text response or call a tool.

        Args:
            state (ChatState): The current state of the conversation.
            config (dict, optional): Configuration dictionary containing 'thread_id'.

        Returns:
            dict: A dictionary containing the new message to append to the state.
        """
        configurable = config.get("configurable", {})
        user_id = configurable.get("user_id")
        thread_id = configurable.get("thread_id")
        if not user_id or not thread_id:
            raise ValueError("Both user_id and thread_id are required.")

        # 1. Fetch Long Term Memory
        items = store.search(("user", user_id, "details"))
        if items:
            user_details_content = "\n".join(it.value.get("data", "") for it in items)
        else:
            user_details_content = "(empty)"

        # 2. Fetch available files to inform the LLM about RAG capabilities
        if thread_id:
            uploaded_files = document_repository.get_thread_file_names(str(thread_id))
        else:
            uploaded_files = []
        if uploaded_files:
            files_context = f"Available documents for RAG: {', '.join(uploaded_files)}"
        else:
            files_context = "No PDF documents uploaded yet."

        latest_human_index = next(
            (
                index
                for index in range(len(state["messages"]) - 1, -1, -1)
                if isinstance(state["messages"][index], HumanMessage)
            ),
            None,
        )
        if latest_human_index is None:
            raise ValueError("Chat state must contain a human message.")

        latest_human = state["messages"][latest_human_index]
        current_turn_messages = state["messages"][latest_human_index:]
        history_messages = _load_history_messages(
            conversation_history_repository=conversation_history_repository,
            embeddings=embeddings,
            user_id=str(user_id),
            thread_id=str(thread_id),
            query=str(latest_human.content),
        )

        # 4. Construct the System Message with dynamic context
        system_message = SystemMessage(
            content=SYSTEM_PROMPT_TEMPLATE.format(
                user_details_content=user_details_content,
                summary_context=(
                    "Relevant conversation history is supplied below."
                    if history_messages
                    else "No relevant previous conversation history."
                ),
                files_context=files_context,
                thread_id=thread_id
            )
        )

        messages = [system_message, *history_messages, *current_turn_messages]
        response = llm_with_tools.invoke(messages, config=config)
        if not response.tool_calls and response.content:
            conversation_history_repository.upsert_turn(
                user_id=str(user_id),
                thread_id=str(thread_id),
                turn_id=str(latest_human.id or uuid.uuid4()),
                user_content=str(latest_human.content),
                assistant_content=_message_content(response),
                embedding=embeddings.embed_query(
                    f"User: {latest_human.content}\nAssistant: {_message_content(response)}"
                ),
            )
        return {"messages": [response]}
    return long_term_memory, chat_node


def _message_content(message: AIMessage) -> str:
    """Convert provider content blocks into text for history indexing."""
    if isinstance(message.content, str):
        return message.content
    return "\n".join(
        block.get("text", "")
        for block in message.content
        if isinstance(block, dict) and block.get("text")
    )


def _load_history_messages(
    *,
    conversation_history_repository,
    embeddings,
    user_id: str,
    thread_id: str,
    query: str,
) -> list[BaseMessage]:
    """Return five recent turns plus additional semantically related turns."""
    recent = conversation_history_repository.recent_turns(
        user_id=user_id,
        thread_id=thread_id,
        limit=RECENT_HISTORY_TURNS,
    )
    semantic = conversation_history_repository.search_turns(
        query_embedding=embeddings.embed_query(query),
        user_id=user_id,
        thread_id=thread_id,
        match_count=LONG_CONTEXT_MATCH_LIMIT,
    )

    turns = {turn["turn_id"]: turn for turn in recent}
    turns.update({turn["turn_id"]: turn for turn in semantic})
    ordered_turns = sorted(
        turns.values(),
        key=lambda turn: turn.get("created_at", ""),
    )

    history_messages: list[BaseMessage] = []
    for turn in ordered_turns:
        history_messages.extend(
            [
                HumanMessage(content=turn["user_content"]),
                AIMessage(content=turn["assistant_content"]),
            ]
        )
    return history_messages

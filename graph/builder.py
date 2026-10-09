"""
Builds the chatbot graph
"""
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition

from graph.state import ChatState
from graph.routing import is_long_chat
from graph.nodes import create_graph_nodes


def build_chatbot(
    *,
    llm,
    store,
    tools,
    checkpointer,
    memory_llm,
    document_repository,
    embeddings,
    conversation_history_repository,
):
    """Builds the chatbot graph with the provided components."""
    # Create a generic ToolNode that executes the tools when called by the graph
    tool_node = ToolNode(tools)

    long_term_memory, chat_node = create_graph_nodes(
        llm=llm,
        llm_with_tools=llm.bind_tools(tools),
        memory_llm=memory_llm,
        document_repository=document_repository,
        embeddings=embeddings,
        conversation_history_repository=conversation_history_repository,
    )

    # Define the Graph structure
    graph = StateGraph(ChatState)

    # Nodes
    graph.add_node("chat_node", chat_node)
    graph.add_node("tool_node", tool_node)
    graph.add_node("long_term_memory", long_term_memory)

    # Edges
    graph.add_conditional_edges(
        START,
        is_long_chat,
        {
            "long_chat": "long_term_memory",
            "short_chat": "chat_node",
        },
    )
    graph.add_edge("long_term_memory", "chat_node")
    graph.add_conditional_edges(
        "chat_node",
        tools_condition,
        {
            "tools": "tool_node",
            END: END,
        }
    )
    graph.add_edge("tool_node", "chat_node")

    # Compile the Graph (passes the checkpointer object at the compilation to add it at each steps)
    app = graph.compile(checkpointer=checkpointer, store=store)

    with open("docs/img/langgraph_workflow.png", "wb") as f:
        f.write(app.get_graph().draw_mermaid_png())

    return app

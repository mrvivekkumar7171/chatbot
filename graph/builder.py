"""
Builds the chatbot graph
"""
from langgraph.graph import StateGraph, START
from langgraph.prebuilt import ToolNode, tools_condition

from graph.state import ChatState
from graph.routing import should_summarize
from graph.nodes import create_graph_nodes


def build_chatbot(
    *,
    llm,
    tools,
    checkpointer,
    store,
    memory_extractor,
    document_repository,
):
    """Builds the chatbot graph with the provided components."""
    llm_with_tools = llm.bind_tools(tools)
    # Create a generic ToolNode that executes the tools when called by the graph
    tool_node = ToolNode(tools)

    remember_node, chat_node, summarize_conversation = create_graph_nodes(
        llm=llm,
        llm_with_tools=llm_with_tools,
        memory_extractor=memory_extractor,
        document_repository=document_repository,
    )

    # Define the Graph structure
    graph = StateGraph(ChatState)

    # Nodes
    graph.add_node("chat_node", chat_node)
    graph.add_node("tools", tool_node)
    graph.add_node("remember_node", remember_node)
    graph.add_node("summarize_conversation", summarize_conversation)

    # Edges
    graph.add_conditional_edges(
        START,
        should_summarize,
        {
            "remember_node": "remember_node",
            "summarize_node": "summarize_conversation",
        },
    )
    graph.add_edge("summarize_conversation", "remember_node")
    graph.add_edge("remember_node", "chat_node")

    # tools_condition decides whether to call a tool or END.
    graph.add_conditional_edges("chat_node", tools_condition)

    # Return to chat after tool execution
    graph.add_edge("tools", "chat_node")

    # Compile the Graph (passes the checkpointer object at the compilation to add it at each steps)
    return graph.compile(
        checkpointer=checkpointer,
        store=store,
    )

"""
Streamlit frontend for the LangGraph AI chatbot, providing chat, conversation history, PDF uploads,
document information, tool status, feedback, and human-in-the-loop controls. It connects to the 
backend for chatbot execution, persistent conversations, document processing, and memory.
"""
import uuid

from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.tracers.context import collect_runs
from langgraph.types import Command
from langsmith import Client
import streamlit as st

from backend import chatbot, retrieve_all_threads, ingest_pdf, get_thread_metadata

# =========================== Utilities ===========================
def generate_thread_id()  -> uuid.UUID:
    """
    Generates a new, unique identifier for a chat thread.

    Returns:
        uuid.UUID: A random UUID object.
    """
    return uuid.uuid4()

def add_thread(thread_id: uuid.UUID | str) -> None:
    """
    Adds a thread ID to the session state list if it doesn't already exist.

    Args:
        thread_id (uuid.UUID | str): The thread ID to add.
    """
    if thread_id not in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].append(thread_id)

def reset_chat() -> None:
    """
    Resets the current session state to start a new conversation.
    It generates a new thread ID, adds it to the list, and clears message history.
    """
    thread_id = generate_thread_id()
    st.session_state["thread_id"] = thread_id
    st.query_params["thread_id"] = str(thread_id)
    add_thread(thread_id)
    st.session_state["message_history"] = []

def load_conversation(thread_id: str) -> list:
    """
    Loads the message history for a specific thread from the backend state.

    Args:
        thread_id (str): The ID of the thread to load.

    Returns:
        list: A list of message objects (HumanMessage, AIMessage, etc.).
    """
    # Fetch the state from LangGraph using the thread_id
    state = chatbot.get_state(config={"configurable": {"thread_id": thread_id}})
    return state.values.get("messages", [])


# ======================= Session Initialization ===================
# Session State is a type of dictionary helps to preserve the data until the whole page is reload.

# Initialize 'message_history' to store chat messages for display
if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

# Initialize 'thread_id' for the current active conversation
if "thread_id" not in st.session_state:
    # 1. Check if thread_id exists in URL query params (Handles Page Refresh)
    query_params = st.query_params
    url_thread_id = query_params.get("thread_id")

    if url_thread_id:
        st.session_state["thread_id"] = url_thread_id
        # Reload history immediately if resuming from URL
        msgs = load_conversation(url_thread_id)

        # Convert to UI format immediately so the user sees history on refresh
        temp_history = []
        for msg in msgs:
            if isinstance(msg, HumanMessage):
                temp_history.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage) and msg.content:
                # Filter out empty AI messages (tool calls)
                # and ensure we don't show System/Tool messages
                temp_history.append({"role": "assistant", "content": msg.content})

        st.session_state["message_history"] = temp_history
    else:
        # 2. If no URL param, generate new ID
        new_id = generate_thread_id()
        st.session_state["thread_id"] = new_id
        st.query_params["thread_id"] = str(new_id)

# Initialize 'chat_threads' by fetching existing threads from DB
if "chat_threads" not in st.session_state:
    st.session_state["chat_threads"] = retrieve_all_threads()

# Initialize a key to reset the file uploader widget programmatically
if "file_uploader_key" not in st.session_state:
    st.session_state["file_uploader_key"] = 0

# Ensure current thread is in the chat_threads list
add_thread(st.session_state["thread_id"])

THREAD_KEY = str(st.session_state["thread_id"])

# Fetch metadata (uploaded files info) for the current thread
current_metadata = get_thread_metadata(THREAD_KEY)
files_info = current_metadata.get("files", {})

# Prepare the list of threads for the sidebar (reversed to show newest first)
threads = st.session_state["chat_threads"][::-1]
selected_thread = None

# We define the user_id here so the backend knows which memory namespace to access.
# In a real app, this would come from a login system.
CURRENT_USER_ID = "user_123"

# Base configuration for LangGraph execution
CONFIG = {
    "run_name": "chat_turn",
    "configurable": {
        "thread_id": THREAD_KEY,
        "user_id": CURRENT_USER_ID
    },
    "metadata": {
        "thread_id": THREAD_KEY,
        "user_id": CURRENT_USER_ID,
        "model": "openai/gpt-oss-20b",
        "temperature": 0.7,
        "parser": "StrOutputParser"
    },
    "tags": ["llm app", "report_generation", "summarization"]
}

st.set_page_config(layout="wide", page_title="TheSoftMax Chat", page_icon="💬")
st.title("TheSoftMax")
# ============================ Sidebar ============================
# Button to start a fresh conversation
if st.sidebar.button("New Chat", use_container_width=True):
    reset_chat()
    st.rerun()

# List of uploaded documents with stats
st.sidebar.subheader("Uploaded Documents")

if files_info:
    for fname, stats in files_info.items():
        st.sidebar.text(f"📄 {fname}")
        st.sidebar.caption(f"{stats['chunks']} chunks | {stats['pages']} pages")
else:
    st.sidebar.info("No documents uploaded yet.")

# Document upload section
uploaded_pdf = st.sidebar.file_uploader(
    " ",
    type=["pdf"],
    key=f"uploader_{st.session_state['file_uploader_key']}"
    )

if uploaded_pdf:
    # Prevent re-indexing the same file
    if uploaded_pdf.name in files_info:
        st.sidebar.info(f"`{uploaded_pdf.name}` is already indexed.")
    else:
        with st.sidebar.status("Indexing PDF…", expanded=True) as status_box:
            # Call backend function to process PDF
            summary = ingest_pdf(
                uploaded_pdf.getvalue(),
                thread_id=THREAD_KEY,
                filename=uploaded_pdf.name,
            )

            if summary.get("error") or summary.get("chunks", 0) == 0:
                status_box.update(label="❌ Failed", state="error")
                st.sidebar.error(summary.get("error", "Processing failed."))
            else:
                status_box.update(label="✅ Indexed", state="complete")
                # Increment key to reset uploader widget and re-run to update the file list
                st.session_state["file_uploader_key"] += 1
                st.rerun()

st.sidebar.divider()
st.sidebar.subheader("History")

# Display list of past threads
if not threads:
    st.sidebar.write("No past conversations.")
else:
    for thread_id_ in threads:
        # Highlight the currently active thread
        label = f"➤ {thread_id_}" if str(thread_id_) == THREAD_KEY else str(thread_id_)
        if st.sidebar.button(label, key=f"side-thread-{thread_id_}"):
            selected_thread = thread_id_


# ============================ Main Chat UI ============================

# 1. Render existing chat history
for message in st.session_state["message_history"]:

    # For each role in message_history display the message in the chat window
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 2. Feedback Scoring
if st.session_state.get("last_run_id"):
    # Display thumbs-up/down feedback
    # We use the run_id as part of the key so the widget resets for new responses
    FEEDBACK = st.feedback("thumbs", key=f"feedback_{st.session_state.last_run_id}")

    if FEEDBACK is not None:
        # feedback value is 1 for Thumbs Up, 0 for Thumbs Down
        SCORE = 1 if FEEDBACK == 1 else 0

        # Send feedback to LangSmith linked to the specific run_id
        Client().create_feedback(
            st.session_state.last_run_id,
            key="user_score",
            score=SCORE
        )
        st.toast("Feedback recorded!", icon="📝")
        del st.session_state["last_run_id"]

# 3. Check for Human-in-the-Loop (HITL) Interrupts
snapshot = chatbot.get_state(CONFIG)
pending_interrupt_value = None
if snapshot.next:
    for task in snapshot.tasks:
        if task.interrupts:
            # Extract the interrupt message (e.g., "Approve buying X shares?")
            pending_interrupt_value = task.interrupts[0].value
            break

# 4. Helper Function to Stream Response
def stream_graph_response(input_payload):
    """
    Streams the response from the LangGraph chatbot to the Streamlit UI.
    Handles ToolMessages (displaying status updates) and AIMessages (streaming text).

    Args:
        input_payload (dict or Command): The input to the graph (user message or resume command).

    Returns:
        str: The final aggregated text content from the assistant.
    """
    with st.chat_message("assistant"):
        # Hold the status of tool used by Chatbot
        status_holder = {"box": None}

        def ai_only_stream():
            """
            Internal generator to yield (Stream) text chunks (the assistant messages)
            from the chatbot response to the UI.
            """
            # 1. Wrap execution in collect_runs to capture the trace ID
            with collect_runs() as cb:
                for message_chunk, metadata in chatbot.stream(
                    input_payload,
                    config=CONFIG,
                    stream_mode="messages",
                ):
                    # Filter out the internal node output : LangGraph streams ALL LLM calls.
                    # We only want the 'chat_node' output.
                    if (
                        (metadata.get("langgraph_node") == "summarize_conversation")
                        or (metadata.get("langgraph_node") == "remember_node")
                        ):
                        continue

                    # Handle Tool Execution Updates
                    if isinstance(message_chunk, ToolMessage):
                        tool_name = getattr(message_chunk, "name", "tool")
                        if status_holder["box"] is None:
                            status_holder["box"] = st.status(
                                f"🔧 Using `{tool_name}` …",
                                expanded=True
                                )
                        else:
                            status_holder["box"].update(
                                label=f"🔧 Using `{tool_name}` …",
                                state="running", expanded=True
                                )

                    # Handle AI Text Updates
                    if isinstance(message_chunk, AIMessage):
                        if message_chunk.content:
                            # Yield the content of AIMessage instead of Return for streaming
                            yield message_chunk.content

                # 2. Capture the run_id of the completed generation and store in session state
                if cb.traced_runs:
                    # Find the root run named "chat_turn" explicitly
                    root_run = next(
                        (run for run in cb.traced_runs if run.name == "chat_turn"),
                        None
                    )
                    if root_run:
                        st.session_state.last_run_id = root_run.id
                    else:
                        # Fallback: If "chat_turn" not found, default to the first collected run
                        st.session_state.last_run_id = cb.traced_runs[0].id

        # Write stream to UI and capture final text
        ai_message = st.write_stream(ai_only_stream())

        # Finalize tool status if it was used
        if status_holder["box"] is not None:
            status_holder["box"].update(label="✅ Tool finished", state="complete", expanded=False)

    return ai_message

# 5. Input Handling (User Text or Approval Buttons)
if pending_interrupt_value:
    # HITL MODE (Waiting for user approval)
    st.info(f"⚠️ Action Required: **{pending_interrupt_value}**")

    col1, col2 = st.columns([1, 1])
    decision = None

    with col1:
        if st.button("✅ Approve"):
            decision = "yes"
            st.balloons()
    with col2:
        if st.button("❌ Deny"):
            decision = "no"

    if decision:
        # Add decision to history for UI consistency
        st.session_state["message_history"].append(
                {
                    "role": "user",
                    "content": f"[Decision: {decision}]"
                }
            )

        # Resume the graph execution with the user's decision
        response_content = stream_graph_response(Command(resume=decision))

        # Append assistant's follow-up response
        st.session_state["message_history"].append(
            {
                "role": "assistant",
                "content": response_content
                }
            )
        st.rerun()
else:
    # STANDARD CHAT MODE
    # st.text_input or st.number_input or st.date_input
    user_input = st.chat_input("Type here ...")

    if user_input:
        # Append user message to local history and display
        st.session_state["message_history"].append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.text(user_input)

        # Send message to backend and stream response
        response = stream_graph_response({"messages": [HumanMessage(content=user_input)]})

        # Append assistant response to local history
        st.session_state["message_history"].append({"role": "assistant", "content": response})

        st.rerun() # Reload UI to display approval buttons

# 6. Thread Switching Logic
if selected_thread:
    st.session_state["thread_id"] = selected_thread
    st.query_params["thread_id"] = str(selected_thread)

    # Reload full conversation history from backend
    messages = load_conversation(selected_thread)

    # Convert LangChain messages to UI-friendly dict format
    temp_messages = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            temp_messages.append({"role": "user", "content": msg.content})
        elif isinstance(msg, AIMessage) and msg.content:
             # Only show AI messages that have actual content (hides internal tool calls)
            temp_messages.append({"role": "assistant", "content": msg.content})

    st.session_state["message_history"] = temp_messages

    # Trigger reload to fetch new metadata for this thread
    st.rerun()

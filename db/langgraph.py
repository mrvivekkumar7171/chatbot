"""
Database setup and management for LangGraph.
"""
import psycopg

from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.store.postgres import PostgresStore

from config.settings import DATABASE_URL

def setup_langgraph_database() -> None:
    """Create or update the LangGraph checkpoint and store tables."""
    try:
        # Run migrations with a dedicated auto-commit connection as PostgresSaver.setup()
        # creates indexes concurrently, which CANNOT run in a transaction block.
        with psycopg.connect(DATABASE_URL, autocommit=True) as setup_conn:
            PostgresSaver(setup_conn).setup()

        with PostgresStore.from_conn_string(DATABASE_URL) as storesetup:
            storesetup.setup()

    except psycopg.Error as e:
        print(f"Warning during DB setup (indexes might already exist): {e}")

def create_checkpointer(pool):
    """Create a Postgres-backed LangGraph checkpoint saver."""
    # Initialize the persistent connection pool. We do NOT use 'with ConnectionPool(...) as pool:'
    # because the pool needs to remain open for the lifetime of the application/module.
    return PostgresSaver(pool)

def create_store(pool):
    """Create a Postgres-backed LangGraph long-term memory store."""
    return PostgresStore(pool)

def get_all_thread_ids(checkpointer) -> list:
    """
    Retrieves a list of all unique conversation thread IDs from the database.

    Returns:
        list: A list of thread_id strings.
    """
    all_threads = set()
    # Iterate through all checkpoints to find unique thread IDs from the database
    for checkpoint in checkpointer.list(None):
        cfg = checkpoint.config.get("configurable", {})
        thread_id = cfg.get("thread_id")
        if thread_id:
            all_threads.add(thread_id)

    return list(all_threads)

"""
Database setup and management for LangGraph.
"""
import psycopg

from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.store.postgres import PostgresStore

def create_checkpointer(pool):
    """_summary_

    Args:
        pool (_type_): _description_

    Returns:
        _type_: _description_
    """
    # Initialize the persistent connection pool. We do NOT use 'with ConnectionPool(...) as pool:'
    # because the pool needs to remain open for the lifetime of the application/module.
    return PostgresSaver(pool)

def create_store(pool):
    """_summary_

    Args:
        pool (_type_): _description_

    Returns:
        _type_: _description_
    """
    return PostgresStore(pool)

def setup_langgraph_database(database_url: str) -> None:
    """_summary_

    Args:
        database_url (str): _description_
    """
    try:
        # Run migrations with a dedicated auto-commit connection as PostgresSaver.setup()
        # creates indexes concurrently, which CANNOT run in a transaction block.
        with psycopg.connect(
            database_url,
            autocommit=True,
        ) as setup_conn:
            PostgresSaver(setup_conn).setup()

        with PostgresStore.from_conn_string(database_url) as storesetup:
            storesetup.setup()

    except psycopg.Error as e:
        print(f"Warning during DB setup (indexes might already exist): {e}")

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

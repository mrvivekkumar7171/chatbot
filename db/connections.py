"""
Database connection management using psycopg_pool for PostgreSQL.
"""
from psycopg_pool import ConnectionPool
from config.settings import DATABASE_POOL_MAX_SIZE, DATABASE_URL

def create_connection_pool():
    """Create the shared PostgreSQL connection pool for LangGraph storage."""
    return ConnectionPool(
        conninfo=DATABASE_URL,
        max_size=DATABASE_POOL_MAX_SIZE,
        kwargs={"autocommit": True},
    )

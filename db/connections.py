"""
Database connection management using psycopg_pool for PostgreSQL.
"""
from psycopg_pool import ConnectionPool
from config.settings import DATABASE_URL

def create_connection_pool():
    """_summary_

    Returns:
        _type_: _description_
    """
    return ConnectionPool(
        conninfo=DATABASE_URL,
        max_size=20,
        kwargs={"autocommit": True},
    )

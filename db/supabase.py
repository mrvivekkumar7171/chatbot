"""
Supabase Client for interacting with the Supabase database.
"""
import supabase
from supabase import Client

from config.settings import (
    SUPABASE_URL,
    SUPABASE_SERVICE_KEY,
)

def create_supabase_client() -> Client:
    """_summary_

    Returns:
        Client: _description_
    """
    return supabase.create_client(
        SUPABASE_URL,
        SUPABASE_SERVICE_KEY,
    )

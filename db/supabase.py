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
    """Create an authenticated Supabase client from application settings."""
    return supabase.create_client(
        SUPABASE_URL,
        SUPABASE_SERVICE_KEY,
    )

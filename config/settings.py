"""
Settings for the application.
"""
import os
from dotenv import load_dotenv

load_dotenv()

SHORT_TERM_MEMORY_LIMIT = int(os.getenv("SHORT_TERM_MEMORY_LIMIT", "10"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "4000"))

SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")
WEATHER_KEY = os.getenv("WEATHERSTACK_KEY")
ALPHA_KEY = os.getenv("ALPHA_VANTAGE_KEY")
DATABASE_URL = os.getenv("DATABASE_URL")
SUPABASE_URL = os.getenv("SUPABASE_URL")

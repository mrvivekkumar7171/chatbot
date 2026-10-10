"""
Weather tool for fetching current weather data.
"""
from langchain_core.tools import tool
from services.weather_service import WeatherService

def create_weather_tool():
    """Create the LangChain tool that retrieves current weather."""
    weather_service = WeatherService()
    @tool
    def get_weather_data(city: str) -> dict:
        """Fetch current weather data for a city."""
        return weather_service.get_current_weather(city)

    return get_weather_data

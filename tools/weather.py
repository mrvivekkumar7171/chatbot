"""
Weather tool for fetching current weather data.
"""
from langchain_core.tools import tool
from services.weather_service import WeatherService

def create_weather_tool():
    """_summary_

    Returns:
        _type_: _description_
    """
    weather_service = WeatherService()
    @tool
    def get_weather_data(city: str) -> dict:
        """Fetch current weather data for a city."""
        return weather_service.get_current_weather(city)

    return get_weather_data

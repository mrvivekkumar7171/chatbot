"""
Weather tool for fetching current weather data.
"""
from langchain_core.tools import tool

def create_weather_tool(weather_service):
    """_summary_

    Args:
        weather_service (_type_): _description_

    Returns:
        _type_: _description_
    """
    @tool
    def get_weather_data(city: str) -> dict:
        """Fetch current weather data for a city."""
        return weather_service.get_current_weather(city)

    return get_weather_data

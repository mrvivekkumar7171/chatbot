"""
Weather Service for fetching current weather information.
"""
from clients.weather_api import WeatherAPIClient

class WeatherService:
    """Provide current weather through the configured weather client."""

    def __init__(self):
        """Initialize the weather API client."""
        self.client = WeatherAPIClient()

    def get_current_weather(self, city: str) -> dict:
        """_summary_

        Args:
            city (str): _description_

        Returns:
            dict: _description_
        """
        return self.client.get_current_weather(city)

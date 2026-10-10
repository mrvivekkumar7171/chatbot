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
        """Fetch current weather conditions for a city.

        Args:
            city: City name accepted by the Weatherstack API.

        Returns:
            The weather response returned by Weatherstack.
        """
        return self.client.get_current_weather(city)

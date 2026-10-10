"""
Weather API Client for fetching current weather information.
"""
import requests
from config.settings import HTTP_TIMEOUT_SECONDS, WEATHER_KEY, WEATHERSTACK_URL

class WeatherAPIClient:
    """Client for retrieving current weather from Weatherstack."""
    def get_current_weather(self, city: str) -> dict:
        """Return current weather data for the requested city."""
        url = (
            f"{WEATHERSTACK_URL}"
            f"?access_key={WEATHER_KEY}&query={city}"
        )

        response = requests.get(url, timeout=HTTP_TIMEOUT_SECONDS)
        return response.json()

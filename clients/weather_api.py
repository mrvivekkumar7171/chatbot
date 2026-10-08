"""
Weather API Client for fetching current weather information.
"""
import requests
from config.settings import WEATHER_KEY

class WeatherAPIClient:
    """_summary_
    """
    def get_current_weather(self, city: str) -> dict:
        """_summary_

        Args:
            city (str): _description_

        Returns:
            dict: _description_
        """
        url = (
            "http://api.weatherstack.com/current"
            f"?access_key={WEATHER_KEY}&query={city}"
        )

        response = requests.get(url, timeout=10)
        return response.json()

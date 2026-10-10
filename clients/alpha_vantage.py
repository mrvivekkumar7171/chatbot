"""
Alpha Vantage Client for fetching stock data.
"""
import requests
from config.settings import ALPHA_KEY, ALPHA_VANTAGE_URL, HTTP_TIMEOUT_SECONDS

class AlphaVantageClient:
    """Client for retrieving stock quotes from Alpha Vantage."""
    def get_global_quote(self, symbol: str) -> dict:
        """
        Fetch the latest stock price for a given symbol using Alpha Vantage.

        Args:
            symbol (str): The stock ticker symbol (e.g., 'AAPL', 'MSFT').

        Returns:
            dict: The JSON response from the Alpha Vantage API containing stock data.
        """
        url = (
            f"{ALPHA_VANTAGE_URL}"
            f"?function=GLOBAL_QUOTE"
            f"&symbol={symbol}"
            f"&apikey={ALPHA_KEY}"
        )

        response = requests.get(url, timeout=HTTP_TIMEOUT_SECONDS)
        return response.json()

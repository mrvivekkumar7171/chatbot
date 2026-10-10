"""
Stock Service for fetching stock prices.
"""
from clients.alpha_vantage import AlphaVantageClient

class StockService:
    """Provide stock quotes through the Alpha Vantage client."""

    def __init__(self):
        """Initialize the stock data client."""
        self.client = AlphaVantageClient()

    def get_stock_price(self, symbol: str) -> dict:
        """Fetch the latest quote for a stock symbol.

        Args:
            symbol: Exchange ticker symbol such as ``AAPL``.

        Returns:
            The stock quote response returned by Alpha Vantage.
        """
        return self.client.get_global_quote(symbol)

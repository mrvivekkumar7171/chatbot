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
        """_summary_

        Args:
            symbol (str): _description_

        Returns:
            dict: _description_
        """
        return self.client.get_global_quote(symbol)

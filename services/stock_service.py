"""
Stock Service for fetching stock prices.
"""

class StockService:
    """_summary_
    """
    def __init__(self, client):
        self.client = client

    def get_stock_price(self, symbol: str) -> dict:
        """_summary_

        Args:
            symbol (str): _description_

        Returns:
            dict: _description_
        """
        return self.client.get_global_quote(symbol)

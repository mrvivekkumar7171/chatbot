"""
Stock price tool for fetching the latest stock prices.
"""
from langchain_core.tools import tool
from services.stock_service import StockService

def create_stock_price_tool():
    """_summary_

    Returns:
        _type_: _description_
    """
    stock_service = StockService()

    @tool
    def get_stock_price(symbol: str) -> dict:
        """Fetch the latest stock price for a stock symbol."""
        return stock_service.get_stock_price(symbol)

    return get_stock_price

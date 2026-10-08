"""
Stock price tool for fetching the latest stock prices.
"""
from langchain_core.tools import tool

def create_stock_price_tool(stock_service):
    """_summary_

    Args:
        stock_service (_type_): _description_

    Returns:
        _type_: _description_
    """
    @tool
    def get_stock_price(symbol: str) -> dict:
        """Fetch the latest stock price for a stock symbol."""
        return stock_service.get_stock_price(symbol)

    return get_stock_price

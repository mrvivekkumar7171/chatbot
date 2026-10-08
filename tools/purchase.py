"""
Tool for purchasing stocks.
"""
from langchain_core.tools import tool
from langgraph.types import interrupt


@tool
def purchase_stock(symbol: str, quantity: int) -> dict:
    """
    Simulate purchasing a specific quantity of a stock.
    
    IMPORTANT: This tool implements a Human-in-the-Loop (HITL) workflow.
    It pauses execution to request user confirmation before proceeding.

    Args:
        symbol (str): The stock ticker symbol.
        quantity (int): The number of shares to buy.

    Returns:
        dict: A status message indicating success or cancellation after user input.
    """
    # Trigger an interrupt in the LangGraph workflow and The value passed
    # here is displayed to the user in the frontend.
    decision = interrupt(
        f"Approve buying {quantity} shares of {symbol}?"
    )

    # Based on the value returned from the frontend (via Command(resume=...))
    if isinstance(decision, str) and decision.lower() == "yes":
        return {
            "status": "success",
            "message": (
                f"Purchase order placed for "
                f"{quantity} shares of {symbol}."
            ),
            "symbol": symbol,
            "quantity": quantity,
        }

    return {
        "status": "cancelled",
        "message": (
            f"Purchase of {quantity} shares of "
            f"{symbol} was declined by human."
        ),
        "symbol": symbol,
        "quantity": quantity,
    }

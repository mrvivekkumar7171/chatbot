"""
Registry for managing and creating tools.
"""
def create_tool_registry(
    search,
    calculator,
    get_stock_price,
    purchase_stock,
    get_weather_data,
    rag_tool,
    shell_tool,
    safe_write_code,
):
    """Build the ordered list of tools available to the chat model.

    Args:
        search: Web-search tool.
        calculator: Arithmetic calculator tool.
        get_stock_price: Stock quote tool.
        purchase_stock: Human-confirmed stock purchase tool.
        get_weather_data: Current weather tool.
        rag_tool: Uploaded-document retrieval tool.
        shell_tool: Restricted read-only terminal tool.
        safe_write_code: AST-validated workspace file writer.

    Returns:
        Tools in the order they should be exposed to the chat model.
    """
    return [
        search,
        calculator,
        get_stock_price,
        purchase_stock,
        get_weather_data,
        rag_tool,
        shell_tool,
        safe_write_code,
    ]

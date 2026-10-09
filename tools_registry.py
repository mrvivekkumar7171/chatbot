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
    """_summary_

    Args:
        search (_type_): _description_
        calculator (_type_): _description_
        get_stock_price (_type_): _description_
        purchase_stock (_type_): _description_
        get_weather_data (_type_): _description_
        rag_tool (_type_): _description_
        shell_tool (_type_): Restricted read-only terminal tool.
        safe_write_code (_type_): AST-validated workspace file writer.

    Returns:
        _type_: _description_
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

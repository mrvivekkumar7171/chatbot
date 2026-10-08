"""
Web search tool for performing searches on the web.
"""
from langchain_core.tools import tool


def create_web_search_tool(search_service):
    """_summary_

    Args:
        search_service (_type_): _description_

    Returns:
        _type_: _description_
    """
    @tool
    def search(query: str) -> str:
        """Search the web using DuckDuckGo."""
        return search_service.search(query)

    return search

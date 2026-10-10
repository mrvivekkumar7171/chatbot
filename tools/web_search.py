"""
Web search tool for performing searches on the web.
"""
from langchain_core.tools import tool
from services.search_service import SearchService

def create_web_search_tool():
    """Create the LangChain tool that performs web searches."""
    search_service = SearchService()
    @tool
    def search(query: str) -> str:
        """Search the web using DuckDuckGo."""
        return search_service.search(query)

    return search

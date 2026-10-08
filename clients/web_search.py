"""
Web Search Client for performing web searches using DuckDuckGo."""
from langchain_community.tools import DuckDuckGoSearchRun

class WebSearchClient:
    """_summary_
    """
    def search(self, query: str) -> str:
        """
        Search the web using DuckDuckGo.
        """
        return DuckDuckGoSearchRun(
            region="us-en"
        ).run(query)

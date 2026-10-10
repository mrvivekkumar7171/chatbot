"""
Web Search Client for performing web searches using DuckDuckGo."""
from langchain_community.tools import DuckDuckGoSearchRun

from config.settings import WEB_SEARCH_REGION


class WebSearchClient:
    """Client wrapper around the configured DuckDuckGo search integration."""
    def search(self, query: str) -> str:
        """
        Search the web using DuckDuckGo.
        """
        return DuckDuckGoSearchRun(
            region=WEB_SEARCH_REGION
        ).run(query)

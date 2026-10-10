"""
Search Service for querying indexed documents.
"""
from clients.web_search import WebSearchClient

class SearchService:
    """Provide web search through the configured search client."""

    def __init__(self):
        """Initialize the web search client."""
        self.client = WebSearchClient()

    def search(self, query: str) -> str:
        """Search the web for information matching a user query.

        Args:
            query: Search terms to send to the web search client.

        Returns:
            The search provider's textual result.
        """
        return self.client.search(query)

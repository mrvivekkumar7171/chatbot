"""
Search Service for querying indexed documents.
"""
from clients.web_search import WebSearchClient

class SearchService:
    """_summary_
    """
    def __init__(self):
        self.client = WebSearchClient()

    def search(self, query: str) -> str:
        """_summary_

        Args:
            query (str): _description_

        Returns:
            str: _description_
        """
        return self.client.search(query)

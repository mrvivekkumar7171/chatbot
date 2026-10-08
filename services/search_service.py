"""
Search Service for querying indexed documents.
"""
class SearchService:
    """_summary_
    """
    def __init__(self, client):
        self.client = client

    def search(self, query: str) -> str:
        """_summary_

        Args:
            query (str): _description_

        Returns:
            str: _description_
        """
        return self.client.search(query)

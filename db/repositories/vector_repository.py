"""
Vector Repository for managing vector-based document searches.
"""
class VectorRepository:
    """Search document chunks using Supabase pgvector RPCs."""

    def __init__(self, supabase_client):
        """Initialize the repository with a Supabase client."""
        self.supabase = supabase_client

    def search_document_chunks(
        self,
        query_embedding,
        thread_id: str,
        match_count: int = 4,
    ):
        """_summary_

        Args:
            query_embedding (_type_): _description_
            thread_id (str): _description_
            match_count (int, optional): _description_. Defaults to 4.

        Returns:
            _type_: _description_
        """
        result = self.supabase.rpc(
            "match_document_chunks",
            {
                "query_embedding": query_embedding,
                "match_thread_id": str(thread_id),
                "match_count": match_count,
            },
        ).execute()

        return result.data or []

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
        """Find the most similar document chunks for a thread.

        Args:
            query_embedding: Vector representation of the search query.
            thread_id: Conversation thread used to scope the search.
            match_count: Maximum number of chunks to return.

        Returns:
            A list of matching document chunk records.
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

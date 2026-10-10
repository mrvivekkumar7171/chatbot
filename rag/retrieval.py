"""
RAG Retriever for searching document chunks.
"""
class RAGRetriever:
    """Retrieve semantically relevant document chunks."""

    def __init__(self, embeddings, vector_repository):
        """Initialize retrieval with embedding and vector repositories."""
        self.embeddings = embeddings
        self.vector_repository = vector_repository

    def retrieve(
        self,
        query: str,
        thread_id: str,
        limit: int = 4,
    ):
        """Retrieve semantically similar document chunks.

        Args:
            query: Natural-language query to embed and search.
            thread_id: Conversation thread used to scope the search.
            limit: Maximum number of matching chunks to return.

        Returns:
            A list of matching document chunk records.
        """
        if not thread_id:
            return []

        query_embedding = self.embeddings.embed_query(query)

        return self.vector_repository.search_document_chunks(
            query_embedding=query_embedding,
            thread_id=thread_id,
            match_count=limit,
        )

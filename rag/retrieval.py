"""
RAG Retriever for searching document chunks.
"""
class RAGRetriever:
    """_summary_
    """
    def __init__(self, embeddings, vector_repository):
        self.embeddings = embeddings
        self.vector_repository = vector_repository

    def retrieve(
        self,
        query: str,
        thread_id: str,
        limit: int = 4,
    ):
        """_summary_

        Args:
            query (str): _description_
            thread_id (str): _description_
            limit (int, optional): _description_. Defaults to 4.

        Returns:
            _type_: _description_
        """
        if not thread_id:
            return []

        query_embedding = self.embeddings.embed_query(query)

        return self.vector_repository.search_document_chunks(
            query_embedding=query_embedding,
            thread_id=thread_id,
            match_count=limit,
        )

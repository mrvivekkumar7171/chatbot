"""
RAG Service for managing and querying documents.
"""
import os
from config.settings import RAG_MAX_PAGES_LIMIT

class RAGService:
    """Coordinate document ingestion and semantic retrieval."""

    def __init__(self, ingestion, retriever):
        """Initialize the service with ingestion and retrieval handlers."""
        self.ingestion = ingestion
        self.retriever = retriever

    def ingest_pdf(
        self,
        file_bytes: bytes,
        thread_id: str,
        filename: str,
    ) -> dict:
        """_summary_

        Args:
            file_bytes (bytes): _description_
            thread_id (str): _description_
            filename (str): _description_

        Returns:
            dict: _description_
        """
        return self.ingestion.ingest_pdf(
            file_bytes=file_bytes,
            thread_id=thread_id,
            filename=filename,
        )

    def query(
        self,
        query: str,
        thread_id: str,
    ) -> dict:
        """_summary_

        Args:
            query (str): _description_
            thread_id (str): _description_

        Returns:
            dict: _description_
        """
        if not thread_id:
            return {
                "error": "No document indexed for this chat. Upload a PDF first.",
                "query": query,
            }

        result = self.retriever.retrieve(
            query=query,
            thread_id=thread_id,
            limit=int(RAG_MAX_PAGES_LIMIT),
        )

        context = []
        sources = set()

        for row in result:
            context.append(row["content"])

            if row.get("metadata"):
                source = row["metadata"].get("source")
                if source:
                    sources.add(os.path.basename(source))

        return {
            "query": query,
            "context": context,
            "sources": list(sources),
        }

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
        """Delegate PDF ingestion to the configured ingestion component.

        Args:
            file_bytes: Raw PDF bytes received from the uploader.
            thread_id: Conversation thread that owns the document.
            filename: Original filename used for storage and metadata.

        Returns:
            A summary of the document ingestion operation.
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
        """Retrieve document context and source filenames for a query.

        Args:
            query: Natural-language question about uploaded documents.
            thread_id: Conversation thread used to scope document retrieval.

        Returns:
            A dictionary containing the query, matching context, and sources.
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

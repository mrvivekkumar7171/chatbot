"""
RAG Ingestion for processing and indexing documents.
"""
import os
import tempfile
import uuid

from config.settings import TEXT_SPLITTER_CHUNK_SIZE, TEXT_SPLITTER_OVERLAP

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader


class RAGIngestion:
    """Extract PDF text, create embeddings, and persist document chunks."""

    def __init__(self, supabase_client, embeddings):
        """Initialize ingestion with storage and embedding clients."""
        self.supabase = supabase_client
        self.embeddings = embeddings

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

        Raises:
            ValueError: _description_

        Returns:
            dict: _description_
        """
        if not file_bytes:
            raise ValueError("No bytes received for ingestion.")

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".pdf",
        ) as temp_file:
            temp_file.write(file_bytes)
            temp_path = temp_file.name

        try:
            # Upload original PDF
            self.supabase.storage.from_("documents").upload(
                f"{thread_id}/{filename}",
                file_bytes,
                {
                    "content-type": "application/pdf",
                    "upsert": "true",
                },
            )

            # Extract text
            loader = PyPDFLoader(temp_path)
            docs = loader.load()

            # Split into chunks
            chunks = RecursiveCharacterTextSplitter(
                chunk_size=int(TEXT_SPLITTER_CHUNK_SIZE),
                chunk_overlap=int(TEXT_SPLITTER_OVERLAP),
                separators=["\n\n", "\n", " ", ""],
            ).split_documents(docs)

            if not chunks:
                return {
                    "error": "No text found in PDF. It might be a scanned image or empty.",
                    "filename": filename,
                    "chunks": 0,
                }

            document_id = str(uuid.uuid4())

            # Document metadata
            self.supabase.table("documents").insert({
                "id": document_id,
                "thread_id": str(thread_id),
                "filename": filename,
                "page_count": len(docs),
                "chunk_count": len(chunks),
            }).execute()

            # Embeddings
            texts = [chunk.page_content for chunk in chunks]
            vectors = self.embeddings.embed_documents(texts)

            # Chunks
            chunk_rows = []

            for chunk, vector in zip(chunks, vectors):
                chunk_rows.append({
                    "document_id": document_id,
                    "thread_id": str(thread_id),
                    "content": chunk.page_content,
                    "page_number": chunk.metadata.get("page", 0) + 1,
                    "embedding": vector,
                    "metadata": chunk.metadata,
                })

            self.supabase.table(
                "document_chunks"
            ).insert(chunk_rows).execute()

            return {
                "filename": filename,
                "documents": len(docs),
                "chunks": len(chunks),
            }

        except (IOError, ValueError) as e:
            return {
                "error": f"Failed to process or parse document: {str(e)}",
                "filename": filename,
                "chunks": 0,
            }

        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

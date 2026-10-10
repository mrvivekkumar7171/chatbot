"""Hosted embeddings used by document and conversation retrieval."""
from google import genai
from google.genai import types

from config.settings import (
    EMBEDDING_DIMENSION,
    EMBEDDING_MODEL,
    GOOGLE_API_KEY,
)


class GeminiEmbeddings:
    """Adapter exposing the embedding methods expected by the RAG services."""

    def __init__(self, api_key: str, model: str, dimension: int):
        """Initialize the Gemini client and validate embedding configuration."""
        if not api_key:
            raise ValueError(
                "GOOGLE_API_KEY is required for hosted embeddings. "
                "Configure it without enabling paid billing."
            )
        if dimension <= 0:
            raise ValueError("EMBEDDING_DIMENSION must be a positive integer.")

        self.client = genai.Client(api_key=api_key)
        self.model = model
        self.dimension = dimension

    def _embed(self, texts: list[str], task_type: str) -> list[list[float]]:
        """Create and validate vectors for a batch of texts."""
        if not texts:
            return []

        response = self.client.models.embed_content(
            model=self.model,
            contents=texts,
            config=types.EmbedContentConfig(
                task_type=task_type,
                output_dimensionality=self.dimension,
            ),
        )
        embeddings = response.embeddings or []
        vectors = [embedding.values for embedding in embeddings]

        if len(vectors) != len(texts):
            raise RuntimeError(
                "Gemini returned an unexpected number of embedding vectors."
            )
        if any(len(vector) != self.dimension for vector in vectors):
            raise RuntimeError(
                "Gemini returned vectors with a dimension different from "
                f"EMBEDDING_DIMENSION={self.dimension}."
            )
        return vectors

    def embed_query(self, text: str) -> list[float]:
        """Embed a search query."""
        return self._embed([text], "RETRIEVAL_QUERY")[0]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed document or conversation records for retrieval."""
        return self._embed(texts, "RETRIEVAL_DOCUMENT")


def create_embeddings() -> GeminiEmbeddings:
    """Create the configured hosted embedding client."""
    return GeminiEmbeddings(
        api_key=GOOGLE_API_KEY,
        model=EMBEDDING_MODEL,
        dimension=EMBEDDING_DIMENSION,
    )

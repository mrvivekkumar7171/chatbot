"""
Embeddings for creating vector representations of text.
"""
from langchain_huggingface import HuggingFaceEmbeddings

def create_embeddings():
    """_summary_

    Returns:
        _type_: _description_
    """
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

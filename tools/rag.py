"""
Rag tool
"""
from typing import Optional
from langchain_core.tools import tool


def create_rag_tool(rag_service):
    """_summary_

    Args:
        rag_service (_type_): _description_

    Returns:
        _type_: _description_
    """
    @tool
    def rag_tool(
        query: str,
        thread_id: Optional[str] = None,
    ) -> dict:
        """
        Retrieves context from uploaded PDF documents relevant to the user's query.

        Args:
            query (str): The semantic search query.
            thread_id (str, optional): The ID of the current thread to scope the search.

        Returns:
            dict: Contains the original query, a list of context strings, and source filenames.
        """

        return rag_service.query(
            query=query,
            thread_id=str(thread_id) if thread_id else None,
        )

    return rag_tool

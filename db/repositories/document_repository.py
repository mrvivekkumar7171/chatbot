"""
Repository for managing document-related operations in the database.
"""
class DocumentRepository:
    """Retrieve document metadata and filenames from Supabase."""

    def __init__(self, supabase_client):
        """Initialize the repository with a Supabase client."""
        self.supabase = supabase_client

    def get_thread_file_names(self, thread_id: str) -> list[str]:
        """
        Retrieves filenames stored in Supabase for a thread.
        """
        if not thread_id:
            return []

        result = (
            self.supabase
            .table("documents")
            .select("filename")
            .eq("thread_id", str(thread_id))
            .execute()
        )

        return [row["filename"] for row in (result.data or [])]

    def get_thread_metadata(self, thread_id: str) -> dict:
        """
        Retrieves document metadata from Supabase for a thread.
        """
        if not thread_id:
            return {"files": {}}

        result = (
            self.supabase
            .table("documents")
            .select("filename, page_count, chunk_count, created_at")
            .eq("thread_id", str(thread_id))
            .execute()
        )

        files = {}

        for row in (result.data or []):
            files[row["filename"]] = {
                "pages": row["page_count"],
                "chunks": row["chunk_count"],
                "created_at": row["created_at"],
            }

        return {"files": files}

"""
Persistence for semantically searchable conversation turns.
"""


class ConversationHistoryRepository:
    """Stores and retrieves completed conversation turns."""

    def __init__(self, supabase_client):
        self.supabase = supabase_client

    def upsert_turn(
        self,
        *,
        user_id: str,
        thread_id: str,
        turn_id: str,
        user_content: str,
        assistant_content: str,
        embedding: list[float],
    ) -> None:
        self.supabase.table("conversation_turns").upsert(
            {
                "user_id": user_id,
                "thread_id": thread_id,
                "turn_id": turn_id,
                "user_content": user_content,
                "assistant_content": assistant_content,
                "content": f"User: {user_content}\nAssistant: {assistant_content}",
                "embedding": embedding,
            },
            on_conflict="user_id,thread_id,turn_id",
        ).execute()

    def search_turns(
        self,
        *,
        query_embedding: list[float],
        user_id: str,
        thread_id: str,
        match_count: int,
    ) -> list[dict]:
        result = self.supabase.rpc(
            "match_conversation_turns",
            {
                "query_embedding": query_embedding,
                "match_user_id": user_id,
                "match_thread_id": thread_id,
                "match_count": match_count,
            },
        ).execute()
        return result.data or []

    def recent_turns(
        self,
        *,
        user_id: str,
        thread_id: str,
        limit: int,
    ) -> list[dict]:
        result = (
            self.supabase
            .table("conversation_turns")
            .select(
                "turn_id, user_content, assistant_content, created_at"
            )
            .eq("user_id", user_id)
            .eq("thread_id", thread_id)
            .order("created_at", desc=True)
            .limit(limit)
            .execute()
        )
        return result.data or []

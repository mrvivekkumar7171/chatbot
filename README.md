## Self Modifying Coding Agent
![Workflow](/docs/img/langgraph_workflow.png)

## Server Setup Instructions

For API key : https://www.alphavantage.co/support/#api-key
Supabase pgvector → similarity search

1. Supabase Table Creation for documents and document_chunks
Setup vector (pgvector) in Extensions
- Database → Extensions → vector extension → Enable
Create the document and vector (pgvector) table
- Supabase → SQL Editor → New query
```sql
create table documents (
    id uuid primary key default gen_random_uuid(),
    thread_id text not null,
    filename text not null,
    page_count integer,
    chunk_count integer,
    created_at timestamptz default now()
);
create table document_chunks (
    id bigint generated always as identity primary key,
    document_id uuid references documents(id) on delete cascade,
    thread_id text not null,
    content text not null,
    page_number integer,
    embedding vector(384),
    metadata jsonb default '{}'::jsonb,
    created_at timestamptz default now()
);
```

2. Create the HNSW vector search index for fast similarity search
```sql
create index document_chunks_embedding_idx
on document_chunks
using hnsw (embedding vector_cosine_ops);

-- find the top 4 similar chunks and 384-dimension of HuggingFaceEmbeddings
create or replace function match_document_chunks(
    query_embedding vector(384),
    match_thread_id text,
    match_count int default 4
)
returns table (
    id bigint,
    content text,
    page_number integer,
    metadata jsonb,
    similarity float
)
language sql
stable
as $$
    select
        dc.id,
        dc.content,
        dc.page_number,
        dc.metadata,
        1 - (dc.embedding <=> query_embedding) as similarity
    from document_chunks dc
    where dc.thread_id = match_thread_id
    order by dc.embedding <=> query_embedding
    limit match_count;
$$;
```

## Setup order after a database reset

Follow these steps in order after deleting the Supabase tables and storage
objects. Run each SQL block in the Supabase SQL Editor.

### Step 1: Create document tables

Run the `documents` and `document_chunks` table migration above.

### Step 2: Create the document vector search function and index

Run the `document_chunks_embedding_idx` index and
`match_document_chunks(...)` function migration above.

### Step 3: Create semantic conversation history

The LangGraph checkpointer remains the source of truth for the complete state of
each thread. A separate Supabase table stores one embedded record for each
completed user/assistant turn. It is scoped by both `user_id` and `thread_id`.
The application retrieves the five newest turns on every request and adds
semantic matches only for long-context recall.

Run this migration in Supabase SQL Editor:

```sql
create table conversation_turns (
    id bigint generated always as identity primary key,
    user_id text not null,
    thread_id text not null,
    turn_id text not null,
    user_content text not null,
    assistant_content text not null,
    content text not null,
    embedding vector(384) not null,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now(),
    constraint conversation_turns_user_thread_turn_key
        unique (user_id, thread_id, turn_id)
);

create index conversation_turns_scope_created_idx
on conversation_turns (user_id, thread_id, created_at desc);

create index conversation_turns_embedding_idx
on conversation_turns
using hnsw (embedding vector_cosine_ops);

create or replace function match_conversation_turns(
    query_embedding vector(384),
    match_user_id text,
    match_thread_id text,
    match_count int default 5
)
returns table (
    turn_id text,
    user_content text,
    assistant_content text,
    created_at timestamptz,
    similarity float
)
language sql
stable
as $$
    select
        ct.turn_id,
        ct.user_content,
        ct.assistant_content,
        ct.created_at,
        1 - (ct.embedding <=> query_embedding) as similarity
    from conversation_turns ct
    where ct.user_id = match_user_id
      and ct.thread_id = match_thread_id
    order by ct.embedding <=> query_embedding
    limit match_count;
$$;
```

No existing `documents`, `document_chunks`, LangGraph checkpoint, or
LangGraph store table needs to be altered for this change. The complete
conversation remains in the checkpointer; `conversation_turns` is an additional
semantic index.

The application creates LangGraph checkpoint and store tables automatically
when `backend.py` starts. Do not manually create those tables.

### Step 4: Create the document storage bucket

Supabase Dashboard → Storage → New bucket → documents

### Step 5: Install dependencies when needed

```shell
conda create --name chatbot python=3.14.8
conda activate chatbot
pip install -r requirements.txt
```

### Step 6: Start the application

From the project directory, run:

```shell
streamlit run frontend.py
```

On startup, the application will automatically initialize the LangGraph checkpoint and long-term-memory store tables.

### Step 7: Run in the Docker sandbox

The terminal tool only permits read-only inspection commands, and code writes are restricted to `/app/workspace`. Build and run the container from the project directory:

```powershell
docker build -t self-mod-agent .
docker run --rm -p 8501:8501 --name agent-container `
  -v "${PWD}\workspace:/app/workspace" `
  self-mod-agent
```

On Bash or macOS/Linux, use this equivalent volume mount:

```shell
docker run --rm -p 8501:8501 --name agent-container \
  -v "$(pwd)/workspace:/app/workspace" \
  self-mod-agent
```

Only the `workspace` directory is mounted from the host, so generated files remain persistent without exposing the rest of the host filesystem.

### Step 8: Visit LangSmith

https://smith.langchain.com/

## Features to Add in the Future
1. Image and audio models: multimodal AI
check if there is max terns or not to prevent infinte loop if the model is not able to generate a response. If the model reaches the maximum number of turns, it should stop generating responses and return an appropriate message to the user.
orchestration
2. Add user authentication and authorization
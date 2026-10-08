## Self Modifying Coding Agent

For API key : https://www.alphavantage.co/support/#api-key
Supabase pgvector → similarity search

**Database**
```
Supabase
│
├── PostgreSQL
│   │
│   ├── documents metadata
│   │   ├── filename
│   │   ├── thread_id
│   │   ├── page_count
│   │   └── chunk_count
│   │
│   ├── document_chunks
│   │   ├── content
│   │   ├── page_number
│   │   ├── metadata
│   │   ├── document embedding (pgvector, 384 dimensions)
│   │   └── vector similarity search index function
│   │
│   ├── LangGraph checkpoints
│   │   └── Chat history / conversation state
│   │
│   └── LangGraph Store
│       └── Long-term user memory
│
└── Storage
    └── documents/
        ├── <thread_id>/
        │   ├── <filename 1>.pdf
        │   └── <filename 2>.pdf
        │
        └── <thread_id>/
            └── <filename 1>.pdf
```

**User Upload Files**
```
PDF uploads PDF
   ↓
Streamlit file uploader
   ↓
Backend ingest_pdf()
   ↓
Upload original PDF
   ↓
Supabase Storage
   │
   └── documents/<thread_id>/<filename>.pdf
   ↓
PyPDFLoader
   ↓
Extract PDF text
   ↓
RecursiveCharacterTextSplitter
   ↓
Create text chunks
   ↓
HuggingFaceEmbeddings
   ↓
384-dimensional embeddings
   ↓
extract PDF text
   ↓
split into chunks using document_chunks
   ↓
Supabase PostgreSQL
   │
   ├── documents
   │      └── document metadata
   │
   └── document_chunks
          ├── chunk text
          ├── page number
          ├── metadata
          └── embedding pgvector
```

**Rag_tool**
```text
User question
       ↓
LangGraph chat_node
       ↓
Groq LLM
       ↓
Is this a question about an uploaded PDF?
       │
       ├── No
       │    ↓
       │  Normal response / other tools
       │
       └── Yes
            ↓
         rag_tool
            ↓
         Generate query embedding
            ↓
         384-dimensional vector
            ↓
         Supabase RPC match_document_chunks(...)
            ↓
         HNSW / pgvector similarity search for the thread_id
            ↓
         Top 4 matching chunks
            ↓
         Return relevant context
            ↓
         Groq LLM
            ↓
         Final answer
```

## Setup Instructions
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

3. Create the documents bucket storage
Supabase Dashboard → Storage → New bucket → documents

4. Create a conda environment
```shell
conda create --name chatbot python=3.14.8
conda activate chatbot
```

5. Install dependencies
```shell
pip install -r requirements.txt
```

6. Run the frontend using streamlit
```shell
streamlit run frontend.py
```

7. Visit LangSmith to track and visualize your LangChain applications:
https://smith.langchain.com/

## Features to Add in the Future
1. Use below platform's api key for Low-frequency background tasks such as summarization, labeling/classification chat, metadata generation, extracting entities, rewriting, tagging, etc.
- Google Gemini API
- Groq
2. Image and audio models: multimodal AI
check if there is max terns or not to prevent infinte loop if the model is not able to generate a response. If the model reaches the maximum number of turns, it should stop generating responses and return an appropriate message to the user.
orchestration
3. find do we can add bash tool in python code to run bash commands and get the output. This can be useful for automating tasks, running scripts, and interacting with the system.
4. Self-Modifying Coding Agent
5. Add user authentication and authorization
6. Dockerize the application
7. 
```
app/
│
├── config/
│   └── settings.py
│
├── models/
│   ├── state.py
│   └── schemas.py
│
├── llm/
│   ├── factory.py
│   └── embeddings.py
│
├── db/
│   ├── postgres.py
│   ├── repositories/
│   │   ├── conversations.py
│   │   ├── messages.py
│   │   ├── memories.py
│   │   └── documents.py
│   └── migrations/
│
├── memory/
│   ├── service.py
│   ├── semantic.py
│   └── long_term.py
│
├── rag/
│   ├── ingestion.py
│   ├── retrieval.py
│   └── service.py
│
├── tools/
│   ├── web_search.py
│   ├── calculator.py
│   ├── stocks.py
│   ├── weather.py
│   ├── purchase.py
│   ├── rag.py
│   └── registry.py
│
├── graph/
│   ├── nodes.py
│   ├── routing.py
│   └── builder.py
│
└── frontend/
    └── streamlit_app.py
```
8. dependency injection: Currently modules directly use globals that is created by the backend at import time. This makes testing difficult. run weather tool using fake API etc.
9. conversation summarization
10. message deletion
11. user identity
12. database/application separation
13. Error correction for LangSmith feedback, Summarization and Web search for 


Now i want you to create an detailed design to implement LLM, database, tools and RAG seperation towards modularity. Once it willl done i will do the rest of modularity and other features and improvement. But I will do this in new chat session with you. So, you just draft the step by step design to implement modularity in the current codebase. Also ask the chatbot to perform taks one by one or step by step.
Make a package such that Each tool should have one responsibility.
```
tools/
    __init__.py
    web_search.py
    calculator.py
    stocks.py
    weather.py
    rag.py
    purchase.py
    registry.py
```
Also, i want tools should not directly own infrastructure. So, that i can replace without rewriting the LangGraph workflow.
```
WeatherTool
   ↓
WeatherService
   ↓
Weather API client
```
```
rag_tool
   ↓
RAGService
   ↓
VectorRepository
   ↓
PostgreSQL
```
3. I would also introduce a tool registry having something conceptually like this inside tools/registry.py
```
ALL_TOOLS = [
    web_search_tool,
    calculator_tool,
    stock_price_tool,
    purchase_stock_tool,
    weather_tool,
    rag_tool,
]
```
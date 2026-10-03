# Codebase Archaeologist

Ask questions about any GitHub repository and get answers grounded in its code, with
file-and-line citations. Built as a retrieval-augmented generation (RAG) pipeline with
layered guardrails.

> **Status:** Phase A, core package under construction. The original notebook prototype
> is preserved in [`legacy/`](legacy/) and tagged [`v0-prototype`](../../tree/v0-prototype).

## Architecture

Solid boxes are implemented; dashed boxes are planned.

```mermaid
flowchart LR
    classDef done fill:#d1fae5,stroke:#059669,color:#064e3b
    classDef planned fill:#f3f4f6,stroke:#9ca3af,stroke-dasharray:5 5,color:#374151

    subgraph Clients
        NB["Demo notebook"]
        CLI["CLI"]
        ST["Streamlit app (Phase B)"]
        API["FastAPI (Phase C)"]
    end

    subgraph Indexing["Indexing job (resumable)"]
        GH["GitHub tree listing<br/>path, size, blob SHA<br/>(no clone)"]
        FLT["File filters + size guard<br/>ext / vendored / generated"]
        FET["Fetcher: raw CDN or tarball stream<br/>byte caps, SHA integrity check,<br/>in memory only"]
        RED["Guardrail: secret + PII redaction"]
        CHK["Readers + chunker<br/>line ranges, stable ids"]
        EMB["Embedder<br/>bge-small-en-v1.5 (local)"]
    end

    subgraph Answering["Answering pipeline"]
        GIN["Guardrail: input checks"]
        CON["Condense follow-up<br/>with chat history"]
        RET["Retriever (MMR)"]
        GCTX["Guardrail: relevance gate +<br/>context injection check"]
        LLM["LLM via OpenRouter<br/>(free models, fallback)"]
        GOUT["Guardrail: citation check +<br/>LLM groundedness judge"]
    end

    subgraph Storage["Supabase Postgres"]
        PG[("pgvector: chunks + embeddings<br/>tables: repos, files, index_jobs")]
    end

    APP["Archaeologist facade<br/>index() / ask()"]
    CFG["Config + logging"]

    NB & CLI & ST & API --> APP
    APP --> GH
    APP --> GIN
    GH --> FLT --> FET --> RED --> CHK --> EMB --> PG
    GIN --> CON --> RET --> GCTX --> LLM --> GOUT
    PG --> RET

    class CFG,GH,FLT,FET done
    class NB,CLI,ST,API,APP,RED,CHK,EMB,GIN,CON,RET,GCTX,LLM,GOUT,PG planned
```

### Where data lives

| Data | Location |
|---|---|
| Embeddings, chunk text, line ranges | Supabase Postgres (pgvector) |
| Repos, commit SHAs, per-file blob SHAs, index job progress | Supabase Postgres |
| Raw source files | Not stored: streamed from GitHub into memory during indexing |
| Embedding model weights | Local Hugging Face cache |
| Secrets | Local `.env` (never committed) |

## Roadmap

- **Phase A:** core Python package: ingestion, redaction, chunking, storage, QA chain, guardrails, CLI
- **Phase B:** Streamlit app
- **Phase C:** FastAPI service with background indexing workers, Docker
- **Phase D:** scale-up: tree-sitter chunking, hybrid search + reranking, symbol index, agentic retrieval

## Development

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"
cp .env.example .env   # fill in your keys
.venv/bin/ruff check . && .venv/bin/pytest -q
```

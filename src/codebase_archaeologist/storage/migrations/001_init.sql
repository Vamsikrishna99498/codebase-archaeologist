-- Codebase Archaeologist schema, v1.
--
-- Tables live in a private schema: Supabase's auto-generated REST API only
-- exposes the schemas listed in its settings ("public" by default), so these
-- tables are unreachable with the public anon key. Row-level security with no
-- policies is a second layer: only the owner role (our app) can read or write.

create extension if not exists vector with schema extensions;

create table repos (
    slug            text primary key,
    full_name       text not null,
    ref             text not null,
    indexed_sha     text,
    embedding_model text not null,
    file_count      integer not null default 0,
    chunk_count     integer not null default 0,
    indexed_at      timestamptz,
    created_at      timestamptz not null default now()
);

-- path -> blob SHA of every indexed file; drives incremental re-indexing.
create table files (
    repo     text not null,
    path     text not null,
    blob_sha text not null,
    primary key (repo, path)
);

create table chunks (
    id                text primary key,
    repo              text not null,
    path              text not null,
    blob_sha          text not null,
    language          text not null,
    chunk_index       integer not null,
    start_line        integer not null,
    end_line          integer not null,
    parent_start_line integer not null,
    parent_end_line   integer not null,
    scope             text,
    text              text not null,
    embed_text        text not null,
    token_count       integer not null,
    embedding         extensions.vector(384) not null  -- bge-small-en-v1.5
);

-- Line-range lookups for small-to-big expansion and per-file deletes.
create index chunks_repo_path_lines_idx on chunks (repo, path, start_line);
-- Approximate nearest-neighbour search by cosine distance.
create index chunks_embedding_hnsw_idx on chunks
    using hnsw (embedding extensions.vector_cosine_ops);

create table index_jobs (
    id             text primary key,
    repo           text not null,
    commit_sha     text not null,
    status         text not null default 'queued'
                   check (status in ('queued', 'running', 'succeeded', 'failed')),
    files_total    integer not null default 0,
    files_done     integer not null default 0,
    chunks_written integer not null default 0,
    error          text,
    started_at     timestamptz not null default now(),
    updated_at     timestamptz not null default now()
);
create index index_jobs_repo_started_idx on index_jobs (repo, started_at desc);

alter table repos      enable row level security;
alter table files      enable row level security;
alter table chunks     enable row level security;
alter table index_jobs enable row level security;

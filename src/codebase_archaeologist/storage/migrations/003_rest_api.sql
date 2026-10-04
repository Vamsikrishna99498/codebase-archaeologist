-- HTTPS access through Supabase's REST API (PostgREST) for networks that block
-- Postgres ports. Only the secret service_role may use the schema; anon and
-- authenticated get no grants, and row-level security stays enabled.
-- (The schema must also be listed under Data API -> Exposed schemas.)
--
-- Can be applied by the app over a direct connection, or pasted into the
-- Supabase SQL Editor (it records itself in schema_migrations either way).

do $$
begin
    if exists (select 1 from pg_roles where rolname = 'service_role') then
        grant usage on schema archaeologist to service_role;
        grant select, insert, update, delete on all tables in schema archaeologist
            to service_role;
        alter default privileges in schema archaeologist
            grant select, insert, update, delete on tables to service_role;
    end if;
end
$$;

-- Vector search for the REST backend (mirrors PostgresVectorStore.search).
create or replace function archaeologist.match_chunks(
    p_repo text,
    p_query extensions.vector(384),
    p_count integer,
    p_with_vectors boolean default false
)
returns table (
    id text, repo text, path text, blob_sha text, language text,
    chunk_index integer, start_line integer, end_line integer,
    parent_start_line integer, parent_end_line integer, scope text,
    text text, token_count integer, score double precision, embedding text
)
language sql
stable
-- Keep scanning the HNSW index until p_count rows pass the repo filter.
set hnsw.iterative_scan = relaxed_order
set search_path = archaeologist, extensions, public
as $$
    select c.id, c.repo, c.path, c.blob_sha, c.language,
           c.chunk_index, c.start_line, c.end_line,
           c.parent_start_line, c.parent_end_line, c.scope,
           c.text, c.token_count,
           1 - (c.embedding <=> p_query) as score,
           case when p_with_vectors then c.embedding::text end as embedding
    from archaeologist.chunks c
    where c.repo = p_repo
    order by c.embedding <=> p_query
    limit p_count
$$;

revoke execute on function archaeologist.match_chunks(text, extensions.vector, integer, boolean)
    from public;
do $$
begin
    if exists (select 1 from pg_roles where rolname = 'service_role') then
        grant execute on function
            archaeologist.match_chunks(text, extensions.vector, integer, boolean)
            to service_role;
    end if;
end
$$;

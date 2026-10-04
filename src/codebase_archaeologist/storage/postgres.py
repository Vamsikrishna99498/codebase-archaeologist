"""Supabase Postgres + pgvector backend.

All tables live in the private ``archaeologist`` schema (see migrations/). Two
store classes share one connection pool, since the VectorStore and MetadataStore
interfaces have overlapping method names.
"""

from __future__ import annotations

import uuid
from importlib import resources

import numpy as np
from pgvector.psycopg import register_vector
from psycopg import Connection
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from codebase_archaeologist.ingestion.chunking import build_embed_text
from codebase_archaeologist.log import get_logger
from codebase_archaeologist.schemas import Chunk, IndexJob, RepoRecord, SearchHit

log = get_logger(__name__)

SCHEMA = "archaeologist"
_CHUNK_COLUMNS = (
    "id, repo, path, blob_sha, language, chunk_index, start_line, end_line, "
    "parent_start_line, parent_end_line, scope, text, token_count"
)
# Column types for binary COPY, in _CHUNK_COLUMNS order plus the embedding.
_CHUNK_COPY_TYPES = [
    *["text"] * 5,
    *["int4"] * 5,
    "text",
    "text",
    "int4",
    "vector",
]


def _configure(conn: Connection) -> None:
    conn.execute(f"set search_path to {SCHEMA}, extensions, public")
    register_vector(conn)
    conn.commit()


def create_pool(url: str, *, min_size: int = 1, max_size: int = 4) -> ConnectionPool:
    """Open a pool; runs migrations first so the vector type exists for every connection."""
    migrate(url)
    return ConnectionPool(
        url,
        min_size=min_size,
        max_size=max_size,
        configure=_configure,
        kwargs={"row_factory": dict_row, "connect_timeout": 15},
        open=True,
    )


def migrate(url: str) -> list[str]:
    """Apply pending SQL migrations in order; returns the names applied."""
    applied: list[str] = []
    files = sorted(
        (f for f in resources.files(__package__).joinpath("migrations").iterdir()),
        key=lambda f: f.name,
    )
    with Connection.connect(url, connect_timeout=15) as conn:
        conn.execute(f"create schema if not exists {SCHEMA}")
        conn.execute(
            f"create table if not exists {SCHEMA}.schema_migrations "
            "(name text primary key, applied_at timestamptz not null default now())"
        )
        conn.execute(f"alter table {SCHEMA}.schema_migrations enable row level security")
        done = {r[0] for r in conn.execute(f"select name from {SCHEMA}.schema_migrations")}
        conn.commit()
        for f in files:
            if not f.name.endswith(".sql") or f.name in done:
                continue
            with conn.transaction():
                conn.execute(f"set local search_path to {SCHEMA}, extensions, public")
                conn.execute(f.read_text())
                conn.execute(f"insert into {SCHEMA}.schema_migrations (name) values (%s)", [f.name])
            log.info("Applied migration %s", f.name)
            applied.append(f.name)
    return applied


def _row_to_chunk(row: dict) -> Chunk:
    fields = {k: row[k] for k in Chunk.model_fields if k != "embed_text"}
    embed_text = build_embed_text(
        row["path"], row["start_line"], row["end_line"], row["language"], row["scope"], row["text"]
    )
    return Chunk(**fields, embed_text=embed_text)


class PostgresVectorStore:
    def __init__(self, pool: ConnectionPool) -> None:
        self.pool = pool

    def upsert(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must have the same length")
        if not chunks:
            return
        cols = _CHUNK_COLUMNS + ", embedding"
        names = [c.strip() for c in cols.split(",")]
        updates = ", ".join(f"{c} = excluded.{c}" for c in names if c != "id")
        # Bulk path: COPY into a temp table, then one INSERT ... ON CONFLICT merge.
        # Row-by-row inserts cost a network round trip each (slow over the pooler).
        with self.pool.connection() as conn, conn.transaction(), conn.cursor() as cur:
            cur.execute(
                "create temp table chunks_stage (like chunks including defaults) on commit drop"
            )
            # Binary COPY: vectors travel as 1.5 KB of float32 instead of ~4 KB of text.
            with cur.copy(f"copy chunks_stage ({cols}) from stdin (format binary)") as copy:
                copy.set_types(_CHUNK_COPY_TYPES)
                for c, v in zip(chunks, vectors, strict=True):
                    copy.write_row(
                        [*(getattr(c, n) for n in names[:-1]), np.asarray(v, dtype=np.float32)]
                    )
            cur.execute(
                f"insert into chunks ({cols}) select {cols} from chunks_stage "
                f"on conflict (id) do update set {updates}"
            )

    def delete_files(self, repo: str, paths: list[str]) -> int:
        if not paths:
            return 0
        with self.pool.connection() as conn:
            cur = conn.execute(
                "delete from chunks where repo = %s and path = any(%s)", [repo, paths]
            )
            return cur.rowcount

    def delete_repo(self, repo: str) -> int:
        with self.pool.connection() as conn:
            return conn.execute("delete from chunks where repo = %s", [repo]).rowcount

    def search(
        self, repo: str, vector: list[float], k: int, *, with_vectors: bool = False
    ) -> list[SearchHit]:
        q = np.asarray(vector, dtype=np.float32)
        extra = ", embedding" if with_vectors else ""
        with self.pool.connection() as conn, conn.transaction():
            # pgvector >= 0.8: keep scanning the HNSW index until k rows pass the
            # repo filter (otherwise a filtered query can return too few rows).
            conn.execute("set local hnsw.iterative_scan = relaxed_order")
            rows = conn.execute(
                f"select {_CHUNK_COLUMNS}{extra}, 1 - (embedding <=> %s) as score "
                "from chunks where repo = %s order by embedding <=> %s limit %s",
                [q, repo, q, k],
            ).fetchall()
        hits = [
            SearchHit(
                chunk=_row_to_chunk(r),
                score=float(r["score"]),
                vector=r["embedding"].to_list() if with_vectors else None,
            )
            for r in rows
        ]
        # relaxed_order may return near-ties slightly out of order.
        return sorted(hits, key=lambda h: h.score, reverse=True)

    def get_overlapping(self, repo: str, path: str, start_line: int, end_line: int) -> list[Chunk]:
        with self.pool.connection() as conn:
            rows = conn.execute(
                f"select {_CHUNK_COLUMNS} from chunks "
                "where repo = %s and path = %s and start_line <= %s and end_line >= %s "
                "order by start_line, chunk_index",
                [repo, path, end_line, start_line],
            ).fetchall()
        return [_row_to_chunk(r) for r in rows]

    def count(self, repo: str) -> int:
        with self.pool.connection() as conn:
            return conn.execute(
                "select count(*) as n from chunks where repo = %s", [repo]
            ).fetchone()["n"]


class PostgresMetadataStore:
    def __init__(self, pool: ConnectionPool) -> None:
        self.pool = pool

    # --- repos ---

    def upsert_repo(self, record: RepoRecord) -> None:
        fields = list(RepoRecord.model_fields)
        updates = ", ".join(f"{f} = excluded.{f}" for f in fields if f != "slug")
        placeholders = ", ".join(["%s"] * len(fields))
        with self.pool.connection() as conn:
            conn.execute(
                f"insert into repos ({', '.join(fields)}) values ({placeholders}) "
                f"on conflict (slug) do update set {updates}",
                [getattr(record, f) for f in fields],
            )

    def get_repo(self, slug: str) -> RepoRecord | None:
        with self.pool.connection() as conn:
            row = conn.execute("select * from repos where slug = %s", [slug]).fetchone()
        return RepoRecord(**{k: row[k] for k in RepoRecord.model_fields}) if row else None

    def list_repos(self) -> list[RepoRecord]:
        with self.pool.connection() as conn:
            rows = conn.execute("select * from repos order by slug").fetchall()
        return [RepoRecord(**{k: r[k] for k in RepoRecord.model_fields}) for r in rows]

    def delete_repo(self, slug: str) -> None:
        with self.pool.connection() as conn, conn.transaction():
            for table in ("repos", "files", "index_jobs"):
                key = "slug" if table == "repos" else "repo"
                conn.execute(f"delete from {table} where {key} = %s", [slug])

    # --- files ---

    def get_files(self, repo: str) -> dict[str, str]:
        with self.pool.connection() as conn:
            rows = conn.execute("select path, blob_sha from files where repo = %s", [repo])
            return {r["path"]: r["blob_sha"] for r in rows}

    def upsert_files(self, repo: str, files: dict[str, str]) -> None:
        if not files:
            return
        with self.pool.connection() as conn, conn.cursor() as cur:
            cur.executemany(
                "insert into files (repo, path, blob_sha) values (%s, %s, %s) "
                "on conflict (repo, path) do update set blob_sha = excluded.blob_sha",
                [(repo, p, sha) for p, sha in files.items()],
            )

    def delete_files(self, repo: str, paths: list[str]) -> None:
        if paths:
            with self.pool.connection() as conn:
                conn.execute("delete from files where repo = %s and path = any(%s)", [repo, paths])

    # --- jobs ---

    def create_job(self, repo: str, commit_sha: str) -> IndexJob:
        with self.pool.connection() as conn:
            row = conn.execute(
                "insert into index_jobs (id, repo, commit_sha) values (%s, %s, %s) returning *",
                [uuid.uuid4().hex, repo, commit_sha],
            ).fetchone()
        return IndexJob(**row)

    def update_job(self, job: IndexJob) -> None:
        with self.pool.connection() as conn:
            conn.execute(
                "update index_jobs set status = %s, files_total = %s, files_done = %s, "
                "chunks_written = %s, error = %s, updated_at = now() where id = %s",
                [
                    job.status,
                    job.files_total,
                    job.files_done,
                    job.chunks_written,
                    job.error,
                    job.id,
                ],
            )

    def get_job(self, job_id: str) -> IndexJob | None:
        with self.pool.connection() as conn:
            row = conn.execute("select * from index_jobs where id = %s", [job_id]).fetchone()
        return IndexJob(**row) if row else None

    def latest_job(self, repo: str) -> IndexJob | None:
        with self.pool.connection() as conn:
            row = conn.execute(
                "select * from index_jobs where repo = %s order by started_at desc limit 1", [repo]
            ).fetchone()
        return IndexJob(**row) if row else None

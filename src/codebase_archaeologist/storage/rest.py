"""Supabase over HTTPS (REST API / PostgREST) for networks that block Postgres ports.

Same tables and contract as the direct Postgres backend. Authenticates with the
secret service-role key, which must never be exposed to browsers or committed.
Requires migration 003 and the ``archaeologist`` schema listed under Data API ->
Exposed schemas.
"""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from typing import Any

import httpx

from codebase_archaeologist.errors import ArchaeologistError
from codebase_archaeologist.ingestion.chunking import build_embed_text
from codebase_archaeologist.schemas import Chunk, IndexJob, RepoRecord, SearchHit

SCHEMA = "archaeologist"
_CHUNK_FIELDS = [f for f in Chunk.model_fields if f != "embed_text"]
_PAGE = 1000  # Supabase's default max rows per response
_UPSERT_BATCH = 200
_IN_BATCH = 50  # values per in.(...) filter, keeps URLs short


class RestClient:
    """Thin PostgREST client bound to the archaeologist schema."""

    def __init__(self, url: str, key: str, *, timeout: float = 30.0) -> None:
        headers = {
            "apikey": key,
            "Accept-Profile": SCHEMA,
            "Content-Profile": SCHEMA,
        }
        if key.startswith("eyJ"):  # legacy JWT service_role key
            headers["Authorization"] = f"Bearer {key}"
        self.http = httpx.Client(
            base_url=url.rstrip("/") + "/rest/v1", headers=headers, timeout=timeout
        )

    def close(self) -> None:
        self.http.close()

    def request(self, method: str, path: str, **kwargs: Any) -> httpx.Response:
        resp = self.http.request(method, path, **kwargs)
        if not resp.is_success:
            # PostgREST error bodies are safe to show (no credentials in them).
            raise ArchaeologistError(
                f"Supabase REST {method} {path} failed ({resp.status_code}): {resp.text[:300]}"
            )
        return resp

    def select_all(self, table: str, params: dict[str, str]) -> list[dict]:
        """GET with pagination past the per-response row cap."""
        rows: list[dict] = []
        while True:
            headers = {"Range-Unit": "items", "Range": f"{len(rows)}-{len(rows) + _PAGE - 1}"}
            page = self.request("GET", f"/{table}", params=params, headers=headers).json()
            rows.extend(page)
            if len(page) < _PAGE:
                return rows

    def upsert(self, table: str, rows: list[dict], on_conflict: str) -> None:
        for i in range(0, len(rows), _UPSERT_BATCH):
            self.request(
                "POST",
                f"/{table}",
                params={"on_conflict": on_conflict},
                headers={"Prefer": "resolution=merge-duplicates,return=minimal"},
                content=json.dumps(rows[i : i + _UPSERT_BATCH], default=str),
            )

    def delete(self, table: str, params: dict[str, str]) -> int:
        resp = self.request(
            "DELETE", f"/{table}", params=params, headers={"Prefer": "count=exact,return=minimal"}
        )
        return _total(resp)

    def count(self, table: str, params: dict[str, str]) -> int:
        resp = self.request(
            "GET",
            f"/{table}",
            params={**params, "select": "id"},
            headers={"Prefer": "count=exact", "Range-Unit": "items", "Range": "0-0"},
        )
        return _total(resp)


def _total(resp: httpx.Response) -> int:
    # Content-Range: "0-0/42" or "*/42"
    return int(resp.headers.get("content-range", "*/0").rsplit("/", 1)[-1] or 0)


def _in(values: list[str]) -> str:
    """PostgREST in.(...) filter with every value double-quoted and escaped."""
    quoted = ",".join('"' + v.replace("\\", "\\\\").replace('"', '\\"') + '"' for v in values)
    return f"in.({quoted})"


def _vector_literal(v: list[float]) -> str:
    return "[" + ",".join(f"{x:.7g}" for x in v) + "]"


def _row_to_chunk(row: dict) -> Chunk:
    fields = {k: row[k] for k in _CHUNK_FIELDS}
    embed_text = build_embed_text(
        row["path"], row["start_line"], row["end_line"], row["language"], row["scope"], row["text"]
    )
    return Chunk(**fields, embed_text=embed_text)


class RestVectorStore:
    def __init__(self, client: RestClient) -> None:
        self.c = client

    def upsert(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must have the same length")
        rows = [
            {**{k: getattr(ch, k) for k in _CHUNK_FIELDS}, "embedding": _vector_literal(v)}
            for ch, v in zip(chunks, vectors, strict=True)
        ]
        self.c.upsert("chunks", rows, on_conflict="id")

    def delete_files(self, repo: str, paths: list[str]) -> int:
        deleted = 0
        for i in range(0, len(paths), _IN_BATCH):
            deleted += self.c.delete(
                "chunks", {"repo": f"eq.{repo}", "path": _in(paths[i : i + _IN_BATCH])}
            )
        return deleted

    def delete_repo(self, repo: str) -> int:
        return self.c.delete("chunks", {"repo": f"eq.{repo}"})

    def search(
        self, repo: str, vector: list[float], k: int, *, with_vectors: bool = False
    ) -> list[SearchHit]:
        rows = self.c.request(
            "POST",
            "/rpc/match_chunks",
            json={
                "p_repo": repo,
                "p_query": _vector_literal(vector),
                "p_count": k,
                "p_with_vectors": with_vectors,
            },
        ).json()
        hits = [
            SearchHit(
                chunk=_row_to_chunk(r),
                score=float(r["score"]),
                vector=json.loads(r["embedding"]) if with_vectors else None,
            )
            for r in rows
        ]
        return sorted(hits, key=lambda h: h.score, reverse=True)

    def get_overlapping(self, repo: str, path: str, start_line: int, end_line: int) -> list[Chunk]:
        rows = self.c.select_all(
            "chunks",
            {
                "select": ",".join(_CHUNK_FIELDS),
                "repo": f"eq.{repo}",
                "path": f"eq.{path}",
                "start_line": f"lte.{end_line}",
                "end_line": f"gte.{start_line}",
                "order": "start_line,chunk_index",
            },
        )
        return [_row_to_chunk(r) for r in rows]

    def count(self, repo: str) -> int:
        return self.c.count("chunks", {"repo": f"eq.{repo}"})


class RestMetadataStore:
    def __init__(self, client: RestClient) -> None:
        self.c = client

    def upsert_repo(self, record: RepoRecord) -> None:
        self.c.upsert("repos", [record.model_dump(mode="json")], on_conflict="slug")

    def get_repo(self, slug: str) -> RepoRecord | None:
        rows = self.c.select_all("repos", {"slug": f"eq.{slug}"})
        return RepoRecord(**rows[0]) if rows else None

    def list_repos(self) -> list[RepoRecord]:
        return [RepoRecord(**r) for r in self.c.select_all("repos", {"order": "slug"})]

    def delete_repo(self, slug: str) -> None:
        self.c.delete("repos", {"slug": f"eq.{slug}"})
        self.c.delete("files", {"repo": f"eq.{slug}"})
        self.c.delete("index_jobs", {"repo": f"eq.{slug}"})

    def get_files(self, repo: str) -> dict[str, str]:
        rows = self.c.select_all("files", {"select": "path,blob_sha", "repo": f"eq.{repo}"})
        return {r["path"]: r["blob_sha"] for r in rows}

    def upsert_files(self, repo: str, files: dict[str, str]) -> None:
        rows = [{"repo": repo, "path": p, "blob_sha": sha} for p, sha in files.items()]
        self.c.upsert("files", rows, on_conflict="repo,path")

    def delete_files(self, repo: str, paths: list[str]) -> None:
        for i in range(0, len(paths), _IN_BATCH):
            self.c.delete("files", {"repo": f"eq.{repo}", "path": _in(paths[i : i + _IN_BATCH])})

    def create_job(self, repo: str, commit_sha: str) -> IndexJob:
        now = datetime.now(UTC).isoformat()
        row = {
            "id": uuid.uuid4().hex,
            "repo": repo,
            "commit_sha": commit_sha,
            "started_at": now,
            "updated_at": now,
        }
        created = self.c.request(
            "POST", "/index_jobs", json=row, headers={"Prefer": "return=representation"}
        ).json()
        return IndexJob(**created[0])

    def update_job(self, job: IndexJob) -> None:
        self.c.request(
            "PATCH",
            "/index_jobs",
            params={"id": f"eq.{job.id}"},
            json={
                "status": job.status,
                "files_total": job.files_total,
                "files_done": job.files_done,
                "chunks_written": job.chunks_written,
                "error": job.error,
                "updated_at": datetime.now(UTC).isoformat(),
            },
            headers={"Prefer": "return=minimal"},
        )

    def get_job(self, job_id: str) -> IndexJob | None:
        rows = self.c.select_all("index_jobs", {"id": f"eq.{job_id}"})
        return IndexJob(**rows[0]) if rows else None

    def latest_job(self, repo: str) -> IndexJob | None:
        rows = self.c.request(
            "GET",
            "/index_jobs",
            params={"repo": f"eq.{repo}", "order": "started_at.desc", "limit": "1"},
        ).json()
        return IndexJob(**rows[0]) if rows else None

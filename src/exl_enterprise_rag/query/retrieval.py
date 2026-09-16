"""Query plane: permission-aware hybrid retrieval + reranking.

Flow (one query):
    user → roles → allowed departments        (ACL, resolved in Postgres)
    query → bge-m3 embedding                  (dense)
    per-department ANN, in parallel           (partial HNSW indexes)
    + lexical tsvector search                 (BM25-ish branch)
    → reciprocal-rank-fusion merge            (RRF: recall stage, WIDE)
    → cross-encoder rerank                    (precision stage, NARROW)

FUNNEL RULE: retrieve wide (top ~25 by RRF), rerank narrow (top 5-8 for
the prompt). Reranking only what will be shown defeats the stage — the
cross-encoder must see candidates RRF ranked poorly, because rescuing
those is exactly its job.

The ACL filter is structural: every SQL branch carries the department
restriction, so a chunk outside the user's departments can never be
ranked, returned, or leaked — there is no unfiltered code path. The
reranker only ever sees hits that already passed that filter.

The SearchBackend protocol is the vector-DB exit ramp: retrieval logic
depends on the interface, not on pgvector.

Requires: pip install "psycopg[binary]" psycopg-pool pgvector sentence-transformers
Environment: DATABASE_URL
"""

from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Protocol, Sequence

from psycopg import sql
from psycopg_pool import ConnectionPool
from pgvector.psycopg import register_vector

RRF_K = 60  # standard reciprocal-rank-fusion constant


# ------------------------------------------------------------------
# Data types
# ------------------------------------------------------------------

@dataclass
class ChunkHit:
    chunk_id: str
    document_id: str
    source_path: str
    department: str
    heading_path: str
    content: str
    dense_score: float | None = None
    dense_rank: int | None = None
    lexical_rank: int | None = None
    rrf_score: float = 0.0
    rerank_score: float | None = None

    def trace(self) -> dict:
        """Compact provenance dict for query_traces.retrieval."""
        return {
            "chunk_id": str(self.chunk_id),
            "department": self.department,
            "dense_score": self.dense_score,
            "dense_rank": self.dense_rank,
            "lexical_rank": self.lexical_rank,
            "rrf_score": round(self.rrf_score, 6),
            "rerank_score": (round(self.rerank_score, 6)
                             if self.rerank_score is not None else None),
        }


# ------------------------------------------------------------------
# Backend interface (the vector-DB exit ramp)
# ------------------------------------------------------------------

class SearchBackend(Protocol):
    def dense_search(self, embedding: Sequence[float], department: str,
                     k: int) -> list[ChunkHit]: ...

    def lexical_search(self, query_text: str, departments: list[str],
                       k: int) -> list[ChunkHit]: ...


class PgVectorStore:
    """pgvector implementation. One dense query per department so each
    hits its own partial HNSW index (department is inlined as a literal —
    a bound parameter would defeat partial-index matching)."""

    def __init__(self, pool: ConnectionPool, emb_table: str):
        self.pool = pool
        self.emb_table = sql.Identifier(emb_table)

    def dense_search(self, embedding, department: str, k: int) -> list[ChunkHit]:
        query = sql.SQL("""
            SELECT c.id, c.document_id, d.source_path, c.department,
                   c.heading_path, c.content,
                   1 - (e.embedding <=> %s::vector) AS score
            FROM {emb} e
            JOIN chunks c    ON c.id = e.chunk_id AND c.status = 'active'
            JOIN documents d ON d.id = c.document_id
            WHERE e.department = {dept}
            ORDER BY e.embedding <=> %s::vector
            LIMIT %s
        """).format(emb=self.emb_table, dept=sql.Literal(department))
        with self.pool.connection() as conn:
            register_vector(conn)
            rows = conn.execute(query, (list(embedding), list(embedding), k)).fetchall()
        return [
            ChunkHit(chunk_id=r[0], document_id=r[1], source_path=r[2],
                     department=r[3], heading_path=r[4], content=r[5],
                     dense_score=float(r[6]))
            for r in rows
        ]

    def lexical_search(self, query_text: str, departments: list[str],
                       k: int) -> list[ChunkHit]:
        with self.pool.connection() as conn:
            rows = conn.execute("""
                SELECT c.id, c.document_id, d.source_path, c.department,
                       c.heading_path, c.content
                FROM chunks c
                JOIN documents d ON d.id = c.document_id
                WHERE c.status = 'active'
                  AND c.department = ANY(%s)
                  AND c.tsv @@ plainto_tsquery('english', %s)
                ORDER BY ts_rank(c.tsv, plainto_tsquery('english', %s)) DESC
                LIMIT %s
            """, (departments, query_text, query_text, k)).fetchall()
        return [
            ChunkHit(chunk_id=r[0], document_id=r[1], source_path=r[2],
                     department=r[3], heading_path=r[4], content=r[5])
            for r in rows
        ]


# ------------------------------------------------------------------
# ACL resolution
# ------------------------------------------------------------------

def departments_for_user(pool: ConnectionPool, email: str) -> list[str]:
    """user → roles → departments. Empty list = user retrieves nothing."""
    with pool.connection() as conn:
        rows = conn.execute("""
            SELECT DISTINCT rda.department
            FROM users u
            JOIN user_roles ur  ON ur.user_id = u.id
            JOIN role_department_access rda ON rda.role_id = ur.role_id
            WHERE u.email = %s
        """, (email,)).fetchall()
    return [r[0] for r in rows]


def resolve_user(pool: ConnectionPool, email: str):
    """email → (user_id, departments).

    Returns (None, []) for an unknown email — the caller decides whether
    that is a hard error (CLI) or an empty-retrieval refusal (service).
    A KNOWN user with no roles returns (user_id, []): identity resolved,
    zero entitlements, retrieval returns nothing, floor refuses. That
    distinction matters for the audit trail — the trace should carry the
    user_id even when the answer is a refusal.
    """
    with pool.connection() as conn:
        row = conn.execute(
            "SELECT id FROM users WHERE email = %s", (email,)
        ).fetchone()
        if row is None:
            return None, []
        user_id = row[0]
        rows = conn.execute("""
            SELECT DISTINCT rda.department
            FROM user_roles ur
            JOIN role_department_access rda ON rda.role_id = ur.role_id
            WHERE ur.user_id = %s
        """, (user_id,)).fetchall()
    return user_id, [r[0] for r in rows]


def departments_for_role(pool: ConnectionPool, role_name: str) -> list[str]:
    """Demo helper: departments granted to a role directly."""
    with pool.connection() as conn:
        rows = conn.execute("""
            SELECT rda.department
            FROM roles r
            JOIN role_department_access rda ON rda.role_id = r.id
            WHERE r.name = %s
        """, (role_name,)).fetchall()
    return [r[0] for r in rows]


# ------------------------------------------------------------------
# RRF merge
# ------------------------------------------------------------------

def rrf_merge(dense_pools: list[list[ChunkHit]],
              lexical: list[ChunkHit], top_k: int) -> list[ChunkHit]:
    """Reciprocal rank fusion.

    Dense hits from ALL departments are first pooled and re-ranked by
    score into one global dense ranking (per-department ranks aren't
    comparable — rank 1 of a weak department shouldn't outweigh rank 5
    of a strong one; cosine scores from one model are comparable).
    """
    merged: dict[str, ChunkHit] = {}

    dense_all = sorted(
        (h for pool in dense_pools for h in pool),
        key=lambda h: h.dense_score or 0.0, reverse=True,
    )
    for rank, hit in enumerate(dense_all, start=1):
        hit.dense_rank = rank
        hit.rrf_score += 1.0 / (RRF_K + rank)
        merged[hit.chunk_id] = hit

    for rank, hit in enumerate(lexical, start=1):
        if hit.chunk_id in merged:
            merged[hit.chunk_id].lexical_rank = rank
            merged[hit.chunk_id].rrf_score += 1.0 / (RRF_K + rank)
        else:
            hit.lexical_rank = rank
            hit.rrf_score += 1.0 / (RRF_K + rank)
            merged[hit.chunk_id] = hit

    return sorted(merged.values(), key=lambda h: h.rrf_score,
                  reverse=True)[:top_k]


# ------------------------------------------------------------------
# Retriever
# ------------------------------------------------------------------

class Retriever:
    def __init__(self, pool: ConnectionPool, backend: SearchBackend,
                 embed_fn):
        """embed_fn: str -> Sequence[float] (normalized query embedding)."""
        self.pool = pool
        self.backend = backend
        self.embed_fn = embed_fn

    def retrieve(self, query_text: str, allowed_departments: list[str],
                 k_per_branch: int = 25, top_k: int = 25
                 ) -> tuple[list[ChunkHit], dict]:
        """Returns (hits, timing) — timing feeds query_traces.latency_ms."""
        if not allowed_departments:
            return [], {"embed_ms": 0, "search_ms": 0}

        t0 = time.perf_counter()
        qvec = self.embed_fn(query_text)
        t_embed = time.perf_counter()

        with ThreadPoolExecutor(max_workers=len(allowed_departments)) as ex:
            dense_futs = [
                ex.submit(self.backend.dense_search, qvec, dept, k_per_branch)
                for dept in allowed_departments
            ]
            lex_fut = ex.submit(self.backend.lexical_search, query_text,
                                allowed_departments, k_per_branch)
            dense_pools = [f.result() for f in dense_futs]
            lexical = lex_fut.result()
        t_search = time.perf_counter()

        hits = rrf_merge(dense_pools, lexical, top_k)
        timing = {
            "embed_ms": round((t_embed - t0) * 1000),
            "search_ms": round((t_search - t_embed) * 1000),
        }
        return hits, timing


# ------------------------------------------------------------------
# CLI demo: same question, two roles — ACL + funnel acceptance test
# ------------------------------------------------------------------

RETRIEVE_WIDE = 25   # candidates RRF hands to the reranker


def _build(pool):
    with pool.connection() as conn:
        row = conn.execute(
            "SELECT model_name, table_name FROM embedding_models "
            "WHERE is_active").fetchone()
    if row is None:
        raise SystemExit("No active embedding model in registry.")
    model_name, emb_table = row

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name)

    def embed_fn(text: str):
        return model.encode([text], normalize_embeddings=True)[0]

    return Retriever(pool, PgVectorStore(pool, emb_table), embed_fn)


def main() -> None:
    import argparse
    p = argparse.ArgumentParser(description="Hybrid ACL-filtered retrieval")
    p.add_argument("query")
    p.add_argument("--role", action="append", required=True,
                   help="Run the query as this role (repeatable to compare).")
    p.add_argument("--top-k", type=int, default=8,
                   help="Final hits after reranking (the prompt budget).")
    p.add_argument("--no-rerank", action="store_true",
                   help="Skip the cross-encoder (compare RRF-only ordering).")
    args = p.parse_args()

    dsn = os.environ.get("DATABASE_URL")
    if not dsn:
        raise SystemExit("Set DATABASE_URL")
    pool = ConnectionPool(dsn, min_size=1, max_size=10, open=True)
    retriever = _build(pool)

    reranker = None
    if not args.no_rerank:
        from src.exl_enterprise_rag.query.rerank import Reranker
        reranker = Reranker()   # first run downloads weights (~2GB) — loudly

    for role in args.role:
        depts = departments_for_role(pool, role)
        print(f"\n=== as {role!r} — departments: {sorted(depts)} ===")

        # WIDE: give the reranker the full RRF candidate pool...
        hits, timing = retriever.retrieve(args.query, depts,
                                          top_k=RETRIEVE_WIDE)
        # ...NARROW: the cross-encoder picks what the prompt will see.
        if reranker is not None:
            t0 = time.perf_counter()
            hits = reranker.rerank(args.query, hits, top_k=args.top_k)
            timing["rerank_ms"] = round((time.perf_counter() - t0) * 1000)
        else:
            hits = hits[:args.top_k]

        print(f"    (embed {timing.get('embed_ms', '?')} ms, "
              f"search {timing.get('search_ms', '?')} ms, "
              f"rerank {timing.get('rerank_ms', '-')} ms)")
        if not hits:
            print("    NO RESULTS — nothing retrievable for this role.")
            continue
        for h in hits:
            branches = []
            if h.dense_rank is not None:
                branches.append(f"dense#{h.dense_rank}({h.dense_score:.3f})")
            if h.lexical_rank is not None:
                branches.append(f"lex#{h.lexical_rank}")
            rr = (f"rr={h.rerank_score:+.3f}  "
                  if h.rerank_score is not None else "")
            print(f"  {rr}rrf={h.rrf_score:.4f}  [{h.department}] "
                  f"{h.heading_path[:90]}  <{' + '.join(branches)}>")

    pool.close()


if __name__ == "__main__":
    main()
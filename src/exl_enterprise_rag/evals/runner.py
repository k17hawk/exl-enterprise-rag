"""Eval harness: golden questions → live pipeline → eval_runs/eval_results.

Runs every golden question through EXACTLY the production path
(resolve identity → retrieve wide → rerank narrow → floor → LLM) and
records, per question:

  * outcome        — 'answered' | 'refused_by_floor' | 'refused_by_model'
                     (honest three-way split; the reason Gap 3 needed
                     Gap 2's outcome fix first)
  * refusal_correct— did expect_refusal match what actually happened
  * acl_clean      — every candidate in the WIDE pool (pre-rerank) sits
                     inside the asker's allowed departments. This is the
                     structural ACL claim asserted where it lives: on
                     the candidate pool, not on the final answer.
  * recall_wide / recall_final
                   — expected_doc_paths coverage in the wide pool vs the
                     top-k that reached the prompt. Matching is by PREFIX
                     so folder-level expectations work; tighten seeds to
                     exact paths as the golden set matures.

Usage:
    python -m src.exl_enterprise_rag.evals.runner            # full run
    python -m src.exl_enterprise_rag.evals.runner --retrieval-only
    python -m src.exl_enterprise_rag.evals.runner --category acl

--retrieval-only skips the LLM: outcome is predicted from the floor
alone ('refused_by_model' can never be observed). Free and fast — use
it while tuning retrieval; use full runs before touching the prompt,
max_tokens, or the floor.

Environment: DATABASE_URL (+ ANTHROPIC_API_KEY unless --retrieval-only)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import time
import uuid
from pathlib import Path

from dotenv import load_dotenv

_REPO_ROOT = Path(__file__).resolve().parents[3]
load_dotenv(_REPO_ROOT / ".env", override=True)

from psycopg_pool import ConnectionPool

from src.exl_enterprise_rag.query.prompts import PROMPT_VERSION
from src.exl_enterprise_rag.query.retrieval import (
    PgVectorStore,
    Retriever,
    departments_for_role,
    resolve_user,
)

log = logging.getLogger("evals.runner")

RETRIEVE_WIDE = 25


# ------------------------------------------------------------------
# DB access
# ------------------------------------------------------------------

def load_golden_questions(pool: ConnectionPool,
                          category: str | None) -> list[dict]:
    q = """
        SELECT id, question, ask_as_email, ask_as_role,
               expect_refusal, expected_doc_paths, category
        FROM golden_questions
    """
    params: tuple = ()
    if category:
        q += " WHERE category = %s"
        params = (category,)
    q += " ORDER BY created_at"
    with pool.connection() as conn:
        rows = conn.execute(q, params).fetchall()
    return [
        {"id": r[0], "question": r[1], "ask_as_email": r[2],
         "ask_as_role": r[3], "expect_refusal": r[4],
         "expected_doc_paths": r[5] or [], "category": r[6]}
        for r in rows
    ]


def open_eval_run(pool: ConnectionPool, config: dict) -> uuid.UUID:
    with pool.connection() as conn:
        run_id = conn.execute(
            "INSERT INTO eval_runs (git_commit, prompt_version, config) "
            "VALUES (%s, %s, %s) RETURNING id",
            (_git_commit(), PROMPT_VERSION, json.dumps(config)),
        ).fetchone()[0]
    return run_id


def write_result(pool: ConnectionPool, run_id, question_id,
                 retrieved: list[dict], answer: str | None,
                 metrics: dict) -> None:
    with pool.connection() as conn:
        conn.execute(
            "INSERT INTO eval_results (run_id, question_id, retrieved, "
            "answer, metrics) VALUES (%s, %s, %s, %s, %s)",
            (run_id, question_id, json.dumps(retrieved), answer,
             json.dumps(metrics)),
        )


def close_eval_run(pool: ConnectionPool, run_id, summary: dict) -> None:
    with pool.connection() as conn:
        conn.execute("UPDATE eval_runs SET summary = %s WHERE id = %s",
                     (json.dumps(summary), run_id))


def _git_commit() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=_REPO_ROOT,
                             capture_output=True, text=True, timeout=10)
        return out.stdout.strip() or None
    except Exception:
        return None


# ------------------------------------------------------------------
# Metrics helpers
# ------------------------------------------------------------------

def _path_recall(expected: list[str], got_paths: list[str]) -> float | None:
    """Fraction of expected paths covered. Prefix match: an expectation
    'content/handbook/finance/' is satisfied by any retrieved path under
    it. Returns None when there are no expectations (refusal cases)."""
    if not expected:
        return None
    covered = sum(
        1 for exp in expected
        if any(p == exp or p.startswith(exp) for p in got_paths)
    )
    return covered / len(expected)


def _consume_stream(gen) -> tuple[str, dict]:
    """Drain answer_stream without printing; return (text, info)."""
    parts: list[str] = []
    try:
        while True:
            parts.append(next(gen))
    except StopIteration as stop:
        return "".join(parts), stop.value


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description="Run the golden-question eval")
    p.add_argument("--retrieval-only", action="store_true",
                   help="Skip the LLM; predict outcome from the floor.")
    p.add_argument("--category", help="Only run this category.")
    p.add_argument("--top-k", type=int, default=6,
                   help="Hits that reach the prompt after reranking.")
    args = p.parse_args()

    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )

    dsn = os.environ.get("DATABASE_URL")
    if not dsn:
        raise SystemExit("Set DATABASE_URL")
    pool = ConnectionPool(dsn, min_size=1, max_size=10, open=True)

    questions = load_golden_questions(pool, args.category)
    if not questions:
        raise SystemExit("No golden questions found — run migration 002 "
                         "or check --category.")

    # ---- Build the production pipeline components ----
    with pool.connection() as conn:
        row = conn.execute(
            "SELECT model_name, table_name FROM embedding_models "
            "WHERE is_active").fetchone()
    if row is None:
        raise SystemExit("No active embedding model.")
    emb_model_name, emb_table = row

    from sentence_transformers import SentenceTransformer
    st_model = SentenceTransformer(emb_model_name)
    retriever = Retriever(
        pool, PgVectorStore(pool, emb_table),
        lambda q: st_model.encode([q], normalize_embeddings=True)[0])

    from src.exl_enterprise_rag.query.rerank import Reranker
    reranker = Reranker()

    from src.exl_enterprise_rag.query.answer import (
        RERANK_FLOOR, AnswerService, QueryTraceRepository,
    )
    service = None
    if not args.retrieval_only:
        service = AnswerService(QueryTraceRepository(pool))

    config = {
        "retrieval_only": args.retrieval_only,
        "embedding_model": emb_model_name,
        "answer_model": service.model if service else None,
        "rerank_floor": RERANK_FLOOR,
        "retrieve_wide": RETRIEVE_WIDE,
        "top_k": args.top_k,
        "prompt_version": PROMPT_VERSION,
        "n_questions": len(questions),
    }
    run_id = open_eval_run(pool, config)
    print(f"eval run {run_id} — {len(questions)} questions "
          f"({'retrieval-only' if args.retrieval_only else 'full'})\n")

    # ---- Aggregates ----
    n = 0
    refusal_correct_n = 0
    acl_clean_n = 0
    recall_wide_vals: list[float] = []
    recall_final_vals: list[float] = []
    outcomes: dict[str, int] = {}

    for gq in questions:
        n += 1
        t_start = time.perf_counter()

        # -- identity resolution (mirrors answer.main) --
        if gq["ask_as_email"]:
            user_id, depts = resolve_user(pool, gq["ask_as_email"])
            if user_id is None:
                log.warning("golden question %s: unknown user %r — "
                            "treating as zero entitlements",
                            gq["id"], gq["ask_as_email"])
                depts = []
            asker = gq["ask_as_email"]
        elif gq["ask_as_role"]:
            user_id = None
            depts = departments_for_role(pool, gq["ask_as_role"])
            asker = f"role:{gq['ask_as_role']}"
        else:
            log.warning("golden question %s has no asker — skipping",
                        gq["id"])
            continue

        # -- retrieve WIDE, snapshot the pool, then rerank NARROW --
        wide_hits, timing = retriever.retrieve(
            gq["question"], depts, top_k=RETRIEVE_WIDE)
        wide_paths = [h.source_path for h in wide_hits]
        wide_depts = sorted({h.department for h in wide_hits})
        acl_clean = all(h.department in depts for h in wide_hits)

        t0 = time.perf_counter()
        final_hits = reranker.rerank(gq["question"], list(wide_hits),
                                     top_k=args.top_k)
        timing["rerank_ms"] = round((time.perf_counter() - t0) * 1000)
        final_paths = [h.source_path for h in final_hits]
        best_rerank = max((h.rerank_score or 0.0 for h in final_hits),
                          default=0.0)

        # -- outcome --
        answer_text: str | None = None
        if args.retrieval_only:
            outcome = ("refused_by_floor"
                       if not final_hits or best_rerank < RERANK_FLOOR
                       else "answered")  # predicted; model never consulted
        else:
            answer_text, info = _consume_stream(service.answer_stream(
                gq["question"], final_hits, user_id=user_id,
                embedding_model=emb_model_name, timing=timing))
            outcome = info["outcome"]

        refused = outcome != "answered"
        refusal_correct = refused == gq["expect_refusal"]
        recall_wide = _path_recall(gq["expected_doc_paths"], wide_paths)
        recall_final = _path_recall(gq["expected_doc_paths"], final_paths)

        metrics = {
            "asker": asker,
            "allowed_departments": sorted(depts),
            "pool_departments": wide_depts,
            "acl_clean": acl_clean,
            "outcome": outcome,
            "outcome_predicted": args.retrieval_only,
            "expect_refusal": gq["expect_refusal"],
            "refusal_correct": refusal_correct,
            "best_rerank_score": round(best_rerank, 6),
            "recall_wide": recall_wide,
            "recall_final": recall_final,
            "latency_ms": timing,
            "total_ms": round((time.perf_counter() - t_start) * 1000),
        }
        retrieved = [h.trace() | {"source_path": h.source_path}
                     for h in final_hits]
        write_result(pool, run_id, gq["id"], retrieved, answer_text,
                     metrics)

        # -- aggregates + console line --
        outcomes[outcome] = outcomes.get(outcome, 0) + 1
        refusal_correct_n += refusal_correct
        acl_clean_n += acl_clean
        if recall_wide is not None:
            recall_wide_vals.append(recall_wide)
        if recall_final is not None:
            recall_final_vals.append(recall_final)

        flag = "OK " if (refusal_correct and acl_clean) else "FAIL"
        print(f"[{flag}] ({gq['category'] or '-':>10}) {asker:<24} "
              f"{outcome:<17} rerank={best_rerank:.3f} "
              f"recall={recall_final if recall_final is not None else '-'}  "
              f"{gq['question'][:60]}")
        if not acl_clean:
            print(f"       !! ACL VIOLATION: pool departments {wide_depts} "
                  f"⊄ allowed {sorted(depts)}")

    summary = {
        "n": n,
        "outcomes": outcomes,
        "refusal_accuracy": round(refusal_correct_n / n, 4) if n else None,
        "acl_clean_rate": round(acl_clean_n / n, 4) if n else None,
        "mean_recall_wide": (round(sum(recall_wide_vals) /
                                   len(recall_wide_vals), 4)
                             if recall_wide_vals else None),
        "mean_recall_final": (round(sum(recall_final_vals) /
                                    len(recall_final_vals), 4)
                              if recall_final_vals else None),
    }
    close_eval_run(pool, run_id, summary)

    print(f"\nsummary: {json.dumps(summary, indent=2)}")
    print(f"\nCompare runs:\n"
          f"  SELECT started_at, git_commit, prompt_version, summary\n"
          f"  FROM eval_runs ORDER BY started_at DESC LIMIT 5;")
    pool.close()


if __name__ == "__main__":
    main()

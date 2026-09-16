"""Evaluation harness: run the golden set through the live pipeline,
score it, record it.

Metrics per question:
  * recall_rrf@25    — expected doc reached the RRF candidate pool
  * recall_rerank@k  — expected doc survived reranking into the prompt
                       (prefix match on source_path)
  * refusal_correct  — for expect_refusal questions: did the system
                       refuse? for answerable ones: did it NOT refuse?
  * judge_correct    — LLM-as-judge: does the answer match the
                       expected_answer? (skipped with --retrieval-only)

Modes:
  --retrieval-only   score retrieval + refusal-floor only; no answer
                     generation, no judge — fast and free, run on every
                     change to chunking/retrieval/rerank
  (default)          full pipeline including generation and judging —
                     run before/after prompt or model changes

Every run writes eval_runs (config + summary) and eval_results (per
question), so any two runs are comparable in SQL.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
import uuid

import typer
import yaml
from psycopg_pool import ConnectionPool
from rich.console import Console
from rich.table import Table

from src.exl_enterprise_rag.query.prompts import (
    NOT_FOUND_ANSWER, PROMPT_VERSION,
)
from src.exl_enterprise_rag.query.retrieval import (
    PgVectorStore, Retriever, departments_for_role,
)
from src.exl_enterprise_rag.query.rerank import Reranker

app = typer.Typer(add_completion=False)
console = Console()

RETRIEVE_WIDE = 25
FINAL_K = 6
RERANK_FLOOR = 0.45

JUDGE_MODEL = os.environ.get("RAG_JUDGE_MODEL", "claude-sonnet-5")

JUDGE_PROMPT = """You are grading a RAG system's answer against a \
reference. Reply with ONLY one word: CORRECT or INCORRECT.

CORRECT means: the answer conveys the same substantive facts as the
reference (extra correct detail is fine; different wording is fine).
INCORRECT means: the answer contradicts the reference, misses its core
fact, or answers a different question.

<question>{question}</question>
<reference>{reference}</reference>
<answer>{answer}</answer>"""


# ------------------------------------------------------------------
# Loading the golden set
# ------------------------------------------------------------------

@app.command()
def load(yaml_path: str) -> None:
    """Load/refresh golden questions from YAML (idempotent by question
    text + role)."""
    pool = _pool()
    with open(yaml_path) as f:
        questions = yaml.safe_load(f)
    inserted = updated = 0
    with pool.connection() as conn:
        for q in questions:
            row = conn.execute(
                "SELECT id FROM golden_questions "
                "WHERE question = %s AND ask_as_role = %s",
                (q["question"], q["ask_as_role"]),
            ).fetchone()
            params = (q.get("expected_answer"),
                      q.get("expected_doc_paths") or [],
                      q["category"], bool(q.get("expect_refusal")))
            if row:
                conn.execute(
                    "UPDATE golden_questions SET expected_answer=%s, "
                    "expected_doc_paths=%s, category=%s, expect_refusal=%s "
                    "WHERE id=%s", params + (row[0],))
                updated += 1
            else:
                conn.execute(
                    "INSERT INTO golden_questions (question, ask_as_role, "
                    "expected_answer, expected_doc_paths, category, "
                    "expect_refusal) VALUES (%s, %s, %s, %s, %s, %s)",
                    (q["question"], q["ask_as_role"]) + params)
                inserted += 1
    console.print(f"loaded: {inserted} inserted, {updated} updated")
    pool.close()


# ------------------------------------------------------------------
# Running the evaluation
# ------------------------------------------------------------------

@app.command()
def run(retrieval_only: bool = typer.Option(False, "--retrieval-only"),
        category: str = typer.Option(None, "--category"),
        final_k: int = typer.Option(FINAL_K, "--final-k")) -> None:
    pool = _pool()
    retriever, reranker, emb_model = _build_query_plane(pool)
    answer_service = None
    judge = None
    if not retrieval_only:
        from src.exl_enterprise_rag.query.answer import (
            AnswerService, QueryTraceRepository,
        )
        from anthropic import Anthropic
        answer_service = AnswerService(QueryTraceRepository(pool))
        judge = Anthropic()

    with pool.connection() as conn:
        where = "WHERE category = %s" if category else ""
        params = (category,) if category else ()
        questions = conn.execute(
            f"SELECT id, question, ask_as_role, expected_answer, "
            f"expected_doc_paths, category, expect_refusal "
            f"FROM golden_questions {where} ORDER BY category, question",
            params).fetchall()
    if not questions:
        raise SystemExit("No golden questions loaded — run `load` first.")

    config = {"retrieve_wide": RETRIEVE_WIDE, "final_k": final_k,
              "rerank_floor": RERANK_FLOOR,
              "retrieval_only": retrieval_only,
              "embedding_model": emb_model}
    run_id = _open_run(pool, config)

    per_cat: dict[str, dict[str, list[float]]] = {}
    for (qid, question, role, expected_answer, expected_paths,
         cat, expect_refusal) in questions:
        depts = departments_for_role(pool, role)
        rrf_hits, _ = retriever.retrieve(question, depts,
                                         top_k=RETRIEVE_WIDE)
        hits = reranker.rerank(question, list(rrf_hits), top_k=final_k)
        best = max((h.rerank_score or 0.0 for h in hits), default=0.0)
        refused = (not hits) or best < RERANK_FLOOR

        metrics: dict = {
            "recall_rrf": _recall(rrf_hits, expected_paths),
            "recall_rerank": _recall(hits, expected_paths),
            "best_rerank_score": round(best, 4),
            "refused": refused,
            "refusal_correct": refused == expect_refusal,
        }

        answer_text = None
        if not retrieval_only and not refused and not expect_refusal:
            parts = []
            gen = answer_service.answer_stream(question, hits,
                                               embedding_model=emb_model)
            try:
                while True:
                    parts.append(next(gen))
            except StopIteration:
                pass
            answer_text = "".join(parts)
            metrics["model_refused"] = NOT_FOUND_ANSWER in answer_text
            if expected_answer and not metrics["model_refused"]:
                metrics["judge_correct"] = _judge(
                    judge, question, expected_answer, answer_text)

        _write_result(pool, run_id, qid, hits, answer_text, metrics)
        bucket = per_cat.setdefault(cat, {})
        for key in ("recall_rrf", "recall_rerank", "refusal_correct",
                    "judge_correct"):
            if key in metrics and metrics[key] is not None:
                bucket.setdefault(key, []).append(float(metrics[key]))
        mark = "✓" if metrics["refusal_correct"] and \
            metrics.get("judge_correct", True) and \
            (expect_refusal or metrics["recall_rerank"]) else "✗"
        console.print(f"  {mark} [{cat}] ({role}) {question[:70]}")

    summary = {
        cat: {k: round(sum(v) / len(v), 3) for k, v in b.items()}
        for cat, b in per_cat.items()
    }
    _close_run(pool, run_id, summary)

    table = Table(title=f"Eval Run {str(run_id)[:8]} — "
                        f"prompt {PROMPT_VERSION}")
    table.add_column("Category")
    table.add_column("recall_rrf@25", justify="right")
    table.add_column(f"recall_rerank@{final_k}", justify="right")
    table.add_column("refusal_acc", justify="right")
    table.add_column("judge_correct", justify="right")
    for cat, m in sorted(summary.items()):
        table.add_row(cat,
                      _fmt(m.get("recall_rrf")),
                      _fmt(m.get("recall_rerank")),
                      _fmt(m.get("refusal_correct")),
                      _fmt(m.get("judge_correct")))
    console.print(table)
    pool.close()


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _pool() -> ConnectionPool:
    dsn = os.environ.get("DATABASE_URL")
    if not dsn:
        raise SystemExit("Set DATABASE_URL")
    return ConnectionPool(dsn, min_size=1, max_size=10, open=True)


def _build_query_plane(pool):
    with pool.connection() as conn:
        row = conn.execute(
            "SELECT model_name, table_name FROM embedding_models "
            "WHERE is_active").fetchone()
    if row is None:
        raise SystemExit("No active embedding model.")
    model_name, emb_table = row
    from sentence_transformers import SentenceTransformer
    st = SentenceTransformer(model_name)
    retriever = Retriever(
        pool, PgVectorStore(pool, emb_table),
        lambda q: st.encode([q], normalize_embeddings=True)[0])
    return retriever, Reranker(), model_name


def _recall(hits, expected_paths) -> bool | None:
    """Prefix match: any hit whose source_path starts with any expected
    prefix. None (excluded from averages) when nothing is expected."""
    if not expected_paths:
        return None
    return any(h.source_path.startswith(p)
               for h in hits for p in expected_paths)


def _judge(client, question: str, reference: str, answer: str) -> bool:
    resp = client.messages.create(
        model=JUDGE_MODEL, max_tokens=8,
        messages=[{"role": "user", "content": JUDGE_PROMPT.format(
            question=question, reference=reference, answer=answer)}])
    verdict = "".join(b.text for b in resp.content
                      if b.type == "text").strip().upper()
    return verdict.startswith("CORRECT")


def _open_run(pool, config: dict) -> uuid.UUID:
    commit = None
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"],
                                capture_output=True, text=True,
                                timeout=10).stdout.strip() or None
    except Exception:
        pass
    with pool.connection() as conn:
        return conn.execute(
            "INSERT INTO eval_runs (git_commit, prompt_version, config) "
            "VALUES (%s, %s, %s) RETURNING id",
            (commit, PROMPT_VERSION, json.dumps(config))).fetchone()[0]


def _write_result(pool, run_id, qid, hits, answer_text,
                  metrics: dict) -> None:
    with pool.connection() as conn:
        conn.execute(
            "INSERT INTO eval_results (run_id, question_id, retrieved, "
            "answer, metrics) VALUES (%s, %s, %s, %s, %s) "
            "ON CONFLICT (run_id, question_id) DO UPDATE SET "
            "retrieved=EXCLUDED.retrieved, answer=EXCLUDED.answer, "
            "metrics=EXCLUDED.metrics",
            (run_id, qid, json.dumps([h.trace() for h in hits]),
             answer_text, json.dumps(metrics)))


def _close_run(pool, run_id, summary: dict) -> None:
    with pool.connection() as conn:
        conn.execute("UPDATE eval_runs SET summary=%s WHERE id=%s",
                     (json.dumps(summary), run_id))


def _fmt(v) -> str:
    return "-" if v is None else f"{v:.0%}"


if __name__ == "__main__":
    app()
"""Answer stage: evidence → prompt → streamed LLM answer → trace.

Composes with the query plane:

    retrieve (wide) → rerank (narrow) → [ANSWER STAGE]
        1. no-answer floor: if best rerank score < RERANK_FLOOR,
           return NOT_FOUND without calling the LLM (cheap, safe,
           and immune to the model talking itself into an answer
           from weak evidence)
        2. build versioned prompt (chunks as delimited DATA)
        3. stream generation from Claude
        4. map [n] citations back to chunk ids
        5. write ONE query_traces row: the full story of the request

Requires: pip install anthropic python-dotenv
Environment: DATABASE_URL, ANTHROPIC_API_KEY

Schema note: this version writes two columns that must exist on
query_traces (see migrations/002_identity_and_eval.sql):

    ALTER TABLE query_traces
        ADD COLUMN stop_reason TEXT,
        ADD COLUMN outcome     TEXT;
        -- 'answered' | 'refused_by_floor' | 'refused_by_model'

Outcome semantics (the eval harness counts on these being honest):
    answered          — model produced a grounded answer
    refused_by_floor  — best rerank score < RERANK_FLOOR; the LLM was
                        never called
    refused_by_model  — the LLM was called and returned NOT_FOUND_ANSWER
Collapsing the last one into 'answered' (the old behavior) makes model
refusals invisible in the audit trail and uncountable in evals.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
import uuid
from pathlib import Path

from dotenv import load_dotenv

# ------------------------------------------------------------------
# Load .env BEFORE anything reads os.environ.
#
# override=True is deliberate: a stray `export ANTHROPIC_BASE_URL=
# http://localhost:11434/api` in ~/.bashrc would otherwise silently
# redirect every request to Ollama and produce a confusing 404
# page-not-found. Project-local config wins over inherited shell state.
# ------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[3]
load_dotenv(_REPO_ROOT / ".env", override=True)

from psycopg_pool import ConnectionPool

from src.exl_enterprise_rag.query.prompts import (
    NOT_FOUND_ANSWER,
    PROMPT_VERSION,
    SYSTEM_PROMPT,
    build_user_message,
)

log = logging.getLogger("query.answer")

# Calibrated on the parental-leave two-role test: relevant evidence
# scored 0.795-0.965, topically-adjacent noise scored 0.002-0.113.
RERANK_FLOOR = 0.2

# Real model id. "claude-sonnet-5" does not exist — the API returns a
# 404 not_found_error for it, which looks identical to a routing 404.
DEFAULT_MODEL = os.environ.get("RAG_ANSWER_MODEL", "claude-sonnet-4-5")

DEFAULT_MAX_TOKENS = int(os.environ.get("RAG_ANSWER_MAX_TOKENS", "2048"))

_CITATION_RE = re.compile(r"\[(\d{1,2})\]")


# ------------------------------------------------------------------
# Trace persistence
# ------------------------------------------------------------------

class QueryTraceRepository:
    """Writes the per-request trace row."""

    def __init__(self, pool: ConnectionPool):
        self.pool = pool

    def write(self, *, raw_query: str, rewritten_query: str | None,
              needs_retrieval: bool, retrieval: list[dict],
              prompt_version: str | None, model: str | None,
              embedding_model: str | None,
              input_tokens: int | None, output_tokens: int | None,
              latency_ms: dict, cache_status: str,
              stop_reason: str | None = None,
              outcome: str = "answered",
              user_id=None, message_id=None) -> uuid.UUID:
        with self.pool.connection() as conn:
            trace_id = conn.execute(
                """
                INSERT INTO query_traces
                    (message_id, user_id, raw_query, rewritten_query,
                     needs_retrieval, retrieval, prompt_version, model,
                     embedding_model, input_tokens, output_tokens,
                     latency_ms, cache_status, stop_reason, outcome)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                        %s, %s)
                RETURNING id
                """,
                (message_id, user_id, raw_query, rewritten_query,
                 needs_retrieval, json.dumps(retrieval), prompt_version,
                 model, embedding_model, input_tokens, output_tokens,
                 json.dumps(latency_ms), cache_status,
                 stop_reason, outcome),
            ).fetchone()[0]
        return trace_id


# ------------------------------------------------------------------
# Answer service
# ------------------------------------------------------------------

class AnswerService:
    def __init__(self, traces: QueryTraceRepository,
                 model: str | None = None,
                 rerank_floor: float = RERANK_FLOOR,
                 max_tokens: int = DEFAULT_MAX_TOKENS):
        from anthropic import Anthropic

        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key or api_key == "ollama":
            raise SystemExit(
                "ANTHROPIC_API_KEY is missing or set to the placeholder "
                "'ollama'. Set a real sk-ant-... value in .env."
            )

        # Honor ANTHROPIC_BASE_URL only if it doesn't point at Ollama's
        # native API. That URL is the classic source of a 404 that looks
        # like an auth error but is actually a routing miss.
        base_url = os.environ.get("ANTHROPIC_BASE_URL")
        kwargs: dict = {"api_key": api_key}
        if base_url:
            if "11434" in base_url:
                log.warning(
                    "ignoring ANTHROPIC_BASE_URL=%r (looks like Ollama; "
                    "Anthropic Messages API is not served there). "
                    "Using default api.anthropic.com.", base_url)
            else:
                kwargs["base_url"] = base_url

        self.client = Anthropic(**kwargs)

        # One-line diagnostic. If this shows localhost, something is still
        # hijacking the base URL and you will get a 404 again.
        log.info("anthropic client ready: base_url=%s key=...%s",
                 self.client.base_url, api_key[-4:])
        print(f"[anthropic] base_url={self.client.base_url} "
              f"key=...{api_key[-4:]} model={model or DEFAULT_MODEL}",
              flush=True)

        self.traces = traces
        self.model = model or DEFAULT_MODEL
        self.rerank_floor = rerank_floor
        self.max_tokens = max_tokens

    def answer_stream(self, question: str, hits, *,
                      user_id=None,
                      embedding_model: str | None = None,
                      timing: dict | None = None):
        """Generator: yields answer text chunks as they stream, then
        returns an info dict via StopIteration.value.

        info = {
            "trace_id": uuid,
            "not_found": bool,
            "outcome": 'answered' | 'refused_by_floor' | 'refused_by_model',
            "citations": [...],
            "truncated": bool,
        }
        """
        timing = dict(timing or {})
        best = max((h.rerank_score or 0.0 for h in hits), default=0.0)

        # ---- No-answer floor: refuse cheaply, before the LLM ----
        if not hits or best < self.rerank_floor:
            yield NOT_FOUND_ANSWER
            trace_id = self.traces.write(
                raw_query=question, rewritten_query=None,
                needs_retrieval=True,
                retrieval=[h.trace() for h in hits],
                prompt_version=PROMPT_VERSION, model=None,
                embedding_model=embedding_model,
                input_tokens=None, output_tokens=None,
                latency_ms=timing, cache_status="miss",
                stop_reason=None, outcome="refused_by_floor",
                user_id=user_id,
            )
            return {"trace_id": trace_id, "not_found": True,
                    "outcome": "refused_by_floor",
                    "citations": [], "truncated": False}

        # ---- Generate ----
        user_message = build_user_message(question, hits)
        t0 = time.perf_counter()
        parts: list[str] = []
        with self.client.messages.stream(
            model=self.model,
            max_tokens=self.max_tokens,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}],
        ) as stream:
            for text in stream.text_stream:
                parts.append(text)
                yield text
            final = stream.get_final_message()
        timing["llm_ms"] = round((time.perf_counter() - t0) * 1000)

        # Record why generation stopped. "max_tokens" means the answer
        # is truncated and we want that countable in the traces, not
        # discoverable by users squinting at cut-off sentences.
        stop_reason = getattr(final, "stop_reason", None)
        timing["stop_reason"] = stop_reason
        truncated = stop_reason == "max_tokens"
        if truncated:
            log.warning("answer truncated at max_tokens=%d "
                        "(output_tokens=%s)",
                        self.max_tokens, final.usage.output_tokens)

        answer_text = "".join(parts)

        # ---- Detect a model-side refusal ----
        # The prompt contract says the model replies with NOT_FOUND_ANSWER
        # verbatim when the evidence doesn't answer the question. Models
        # occasionally append a sentence of hedging after it, which is
        # still a refusal semantically — hence startswith, not equality.
        model_refused = answer_text.strip().startswith(NOT_FOUND_ANSWER)
        outcome = "refused_by_model" if model_refused else "answered"

        # ---- Map [n] citations back to chunk ids ----
        cited_numbers = sorted({int(m) for m in
                                _CITATION_RE.findall(answer_text)})
        citations = [
            {"n": n,
             "chunk_id": str(hits[n - 1].chunk_id),
             "heading_path": hits[n - 1].heading_path,
             "source_path": hits[n - 1].source_path}
            for n in cited_numbers if 1 <= n <= len(hits)
        ]

        retrieval_trace = [h.trace() for h in hits]
        for entry in retrieval_trace:
            entry["in_prompt"] = True

        trace_id = self.traces.write(
            raw_query=question, rewritten_query=None,
            needs_retrieval=True, retrieval=retrieval_trace,
            prompt_version=PROMPT_VERSION, model=self.model,
            embedding_model=embedding_model,
            input_tokens=final.usage.input_tokens,
            output_tokens=final.usage.output_tokens,
            latency_ms=timing, cache_status="miss",
            stop_reason=stop_reason, outcome=outcome,
            user_id=user_id,
        )
        return {"trace_id": trace_id, "not_found": model_refused,
                "outcome": outcome,
                "citations": citations, "truncated": truncated}


# ------------------------------------------------------------------
# CLI: the full loop — question + role → cited, streamed answer
# ------------------------------------------------------------------

def main() -> None:
    import argparse
    from src.exl_enterprise_rag.query.retrieval import (
        PgVectorStore, Retriever, departments_for_role, resolve_user,
    )
    from src.exl_enterprise_rag.query.rerank import Reranker

    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )

    p = argparse.ArgumentParser(description="Ask the handbook (end to end)")
    p.add_argument("query")
    p.add_argument("--email",
                   help="Ask as this user (the identity path: "
                        "email → roles → departments; user_id is traced).")
    p.add_argument("--role",
                   help="DEMO ONLY: bypass identity and ask as a bare "
                        "role. No user_id lands in the trace.")
    p.add_argument("--top-k", type=int, default=6)
    args = p.parse_args()

    if not args.email and not args.role:
        p.error("provide --email (or --role for an identity-free demo)")
    if args.email and args.role:
        p.error("--email and --role are mutually exclusive")

    dsn = os.environ.get("DATABASE_URL")
    if not dsn:
        raise SystemExit("Set DATABASE_URL")
    pool = ConnectionPool(dsn, min_size=1, max_size=10, open=True)

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
    reranker = Reranker()
    service = AnswerService(QueryTraceRepository(pool))

    if args.email:
        user_id, depts = resolve_user(pool, args.email)
        if user_id is None:
            raise SystemExit(f"Unknown user: {args.email!r}. "
                             f"Seed users/user_roles first "
                             f"(migrations/002_identity_and_eval.sql).")
        print(f"[{args.email} → departments {sorted(depts)}]\n")
    else:
        user_id = None
        depts = departments_for_role(pool, args.role)
        print(f"[role demo {args.role!r} → departments {sorted(depts)}]\n")

    hits, timing = retriever.retrieve(args.query, depts, top_k=25)
    t0 = time.perf_counter()
    hits = reranker.rerank(args.query, hits, top_k=args.top_k)
    timing["rerank_ms"] = round((time.perf_counter() - t0) * 1000)

    gen = service.answer_stream(args.query, hits,
                                user_id=user_id,
                                embedding_model=emb_model_name,
                                timing=timing)
    info = None
    try:
        while True:
            print(next(gen), end="", flush=True)
    except StopIteration as stop:
        info = stop.value
    print("\n")

    if info and info["citations"]:
        print("Sources:")
        for c in info["citations"]:
            print(f"  [{c['n']}] {c['heading_path']}")
    if info and info.get("truncated"):
        print("\n(answer truncated — raise RAG_ANSWER_MAX_TOKENS to "
              "see the rest)")
    if info:
        print(f"\n(trace {info['trace_id']})")
    pool.close()


if __name__ == "__main__":
    main()
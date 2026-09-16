"""Ingestion pipeline orchestrator.

Stages (each independently testable):

    walk ──► transform (parse→clean→chunk, PURE) ──► plan (hash diff)
         ──► embed (batched, retried) ──► write (per-doc transaction)
         ──► tombstone ──► report

Design rules:
  * Transforms never touch the database; repositories never transform.
  * Documents are processed in WAVES (default 200 docs) so memory stays
    bounded and embedding batches stay GPU-efficient across documents.
  * One file's failure is isolated: logged, recorded in ingestion_errors,
    run continues. The run only reports 'failed' if the pipeline itself
    breaks.
  * Every run is recorded in ingestion_runs with full stats.

Scaling path (documented, not built): when corpus size or ingestion
frequency outgrows a single process, each wave becomes a queue message
and this orchestrator splits into producer (walk+plan) and consumer
(transform+embed+write) workers — Celery or Temporal. The stage seams
below are exactly the message boundaries that split would use.
"""

from __future__ import annotations

import logging
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

from src.exl_enterprise_rag.db.repository import (
    DocumentRepository,
    EmbeddingModelRegistry,
    IngestionRunRepository,
)
from src.exl_enterprise_rag.ingest.chunker import build_chunks, split_into_sections
from src.exl_enterprise_rag.ingest.parser import parse_markdown
from src.exl_enterprise_rag.ingest.shortcodes import clean_shortcodes
from src.exl_enterprise_rag.ingest.titles import is_stub_index, resolve_title
from src.exl_enterprise_rag.ingest.walker import walk_corpus

log = logging.getLogger("pipeline.ingestion")


# ------------------------------------------------------------------
# Transform stage (pure: file in, chunks out, no IO beyond reading)
# ------------------------------------------------------------------

@dataclass(frozen=True)
class TransformResult:
    source_path: str
    department: str
    title: str | None = None
    content_hash: str | None = None
    chunks: tuple = ()
    skip_reason: str | None = None      # 'stub' | 'empty' | None


class DocumentTransformer:
    def __init__(self, includes_dir: Path, min_tokens: int, max_tokens: int):
        self.includes_dir = includes_dir
        self.min_tokens = min_tokens
        self.max_tokens = max_tokens

    def transform(self, source_file) -> TransformResult:
        parsed = parse_markdown(source_file.path)
        body = clean_shortcodes(parsed.body, includes_dir=self.includes_dir)

        if source_file.path.name == "_index.md" and is_stub_index(body):
            return TransformResult(source_file.source_path,
                                   source_file.department,
                                   skip_reason="stub")

        sections = split_into_sections(body)
        title = resolve_title(parsed.title, sections, source_file.path)
        chunks = build_chunks(sections, title=title,
                              min_tokens=self.min_tokens,
                              max_tokens=self.max_tokens)
        if not chunks:
            return TransformResult(source_file.source_path,
                                   source_file.department,
                                   skip_reason="empty")

        return TransformResult(
            source_path=source_file.source_path,
            department=source_file.department,
            title=title,
            content_hash=parsed.content_hash,
            chunks=tuple(chunks),
        )


# ------------------------------------------------------------------
# Embedding stage (batched across documents, retried)
# ------------------------------------------------------------------

class EmbeddingService:
    def __init__(self, model_name: str, expected_dims: int,
                 batch_size: int = 64, max_retries: int = 3):
        from sentence_transformers import SentenceTransformer
        log.info("loading embedding model %s", model_name)
        self.model = SentenceTransformer(model_name)
        self.batch_size = batch_size
        self.max_retries = max_retries

        probe = self.model.encode(["dim probe"], normalize_embeddings=True)
        if probe.shape[1] != expected_dims:
            raise RuntimeError(
                f"model emits {probe.shape[1]} dims, registry says "
                f"{expected_dims} — fix registry/table before ingesting")

    def embed(self, texts: list[str]) -> list:
        out = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            for attempt in range(1, self.max_retries + 1):
                try:
                    out.extend(self.model.encode(
                        batch, normalize_embeddings=True))
                    break
                except Exception:
                    if attempt == self.max_retries:
                        raise
                    wait = 2 ** attempt
                    log.warning("embed batch failed (attempt %d), "
                                "retrying in %ds", attempt, wait)
                    time.sleep(wait)
        return out


# ------------------------------------------------------------------
# Orchestrator
# ------------------------------------------------------------------

@dataclass
class RunStats:
    walked: int = 0
    added: int = 0
    updated: int = 0
    unchanged: int = 0
    skipped_stub: int = 0
    skipped_empty: int = 0
    errored: int = 0
    chunks_written: int = 0
    tombstoned: int = 0
    waves: int = 0
    embed_seconds: float = 0.0
    write_seconds: float = 0.0

    def as_dict(self) -> dict:
        d = self.__dict__.copy()
        d["embed_seconds"] = round(d["embed_seconds"], 1)
        d["write_seconds"] = round(d["write_seconds"], 1)
        return d

    def assert_balanced(self) -> None:
        accounted = (self.added + self.updated + self.unchanged +
                     self.skipped_stub + self.skipped_empty + self.errored)
        assert self.walked == accounted, (
            f"{self.walked - accounted} files unaccounted for")


class IngestionPipeline:
    def __init__(self, *, settings, departments,
                 registry: EmbeddingModelRegistry,
                 runs: IngestionRunRepository,
                 documents: DocumentRepository,
                 embedder: EmbeddingService,
                 transformer: DocumentTransformer):
        self.settings = settings
        self.departments = departments
        self.registry = registry
        self.runs = runs
        self.documents = documents
        self.embedder = embedder
        self.transformer = transformer

    # -- helpers ----------------------------------------------------

    def _git_commit(self) -> str | None:
        try:
            out = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=self.settings.handbook_root,
                capture_output=True, text=True, timeout=10)
            return out.stdout.strip() or None
        except Exception:
            return None

    # -- the run ----------------------------------------------------

    def run(self, *, only_departments: set[str] | None = None,
            force: bool = False, dry_run: bool = False,
            wave_size: int = 200) -> RunStats:
        stats = RunStats()
        files = walk_corpus(self.settings, self.departments,
                            only_departments=only_departments)
        log.info("walk: %d files (filter=%s, force=%s, dry_run=%s)",
                 len(files), sorted(only_departments or []) or "all",
                 force, dry_run)

        run_id = None if dry_run else self.runs.open(self._git_commit())
        seen_paths: list[str] = []

        try:
            for wave_start in range(0, len(files), wave_size):
                wave = files[wave_start:wave_start + wave_size]
                stats.waves += 1
                self._process_wave(wave, stats, seen_paths,
                                   run_id=run_id, force=force,
                                   dry_run=dry_run)
                log.info("wave %d done: +%d added, %d updated, "
                         "%d unchanged, %d errors",
                         stats.waves, stats.added, stats.updated,
                         stats.unchanged, stats.errored)

            if not dry_run and only_departments is None:
                stats.tombstoned = self.documents.tombstone_missing(seen_paths)
                if stats.tombstoned:
                    log.info("tombstoned %d vanished documents",
                             stats.tombstoned)

            stats.walked = len(files)
            stats.assert_balanced()
            if run_id is not None:
                self.runs.close(run_id, "succeeded", stats.as_dict())
            return stats

        except Exception:
            if run_id is not None:
                self.runs.close(run_id, "failed", stats.as_dict())
            raise

    def _process_wave(self, wave, stats: RunStats, seen_paths: list[str],
                      *, run_id, force: bool, dry_run: bool) -> None:
        # 1) transform + plan (hash diff) — collect docs needing writes
        pending: list[TransformResult] = []      # need embed+write
        pending_is_new: list[bool] = []

        for sf in wave:
            try:
                result = self.transformer.transform(sf)
            except Exception as e:
                stats.errored += 1
                log.warning("transform failed: %s: %s", sf.source_path, e)
                if run_id is not None:
                    self.runs.record_error(run_id, sf.source_path, str(e))
                continue

            if result.skip_reason == "stub":
                stats.skipped_stub += 1
                continue
            if result.skip_reason == "empty":
                stats.skipped_empty += 1
                continue

            seen_paths.append(result.source_path)
            existing_hash = self.documents.content_hash(result.source_path)
            if existing_hash == result.content_hash and not force:
                stats.unchanged += 1
                continue

            pending.append(result)
            pending_is_new.append(existing_hash is None)

        if not pending:
            return
        if dry_run:
            for res, is_new in zip(pending, pending_is_new):
                stats.added += is_new
                stats.updated += not is_new
                stats.chunks_written += len(res.chunks)
            return

        # 2) embed the whole wave in one batched pass (GPU-efficient)
        t0 = time.perf_counter()
        texts = [c.content for res in pending for c in res.chunks]
        vectors = self.embedder.embed(texts)
        stats.embed_seconds += time.perf_counter() - t0

        # 3) write per document (atomic each), slicing vectors back out
        t0 = time.perf_counter()
        offset = 0
        for res, is_new in zip(pending, pending_is_new):
            n = len(res.chunks)
            doc_vectors = vectors[offset:offset + n]
            offset += n
            try:
                self.documents.write_document(
                    source_path=res.source_path,
                    title=res.title or "",
                    department=res.department,
                    content_hash=res.content_hash or "",
                    chunks=res.chunks,
                    vectors=doc_vectors,
                )
            except Exception as e:
                stats.errored += 1
                log.warning("write failed: %s: %s", res.source_path, e)
                if run_id is not None:
                    self.runs.record_error(run_id, res.source_path, str(e))
                continue
            stats.added += is_new
            stats.updated += not is_new
            stats.chunks_written += n
        stats.write_seconds += time.perf_counter() - t0
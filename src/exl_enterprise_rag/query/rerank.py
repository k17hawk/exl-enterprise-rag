"""Cross-encoder reranking stage.

Why this stage exists: RRF ranks by fusion arithmetic over branch ranks —
it never reads the chunk. A chunk stuffed with the query's keywords
("Remote.com > Italy" for a parental-leave query) can win lex#1 and float
to the top while the canonical policy section sits at rank 7. A
cross-encoder reads (query, chunk) TOGETHER through one transformer pass
and scores actual relevance, fixing exactly that failure.

Cost model: one forward pass per candidate — expensive per item, so it
runs on the top ~25 RRF candidates only, never on the corpus. This is
the standard retrieve-wide → rerank-narrow funnel.

Model: BAAI/bge-reranker-v2-m3 — same family as the bge-m3 embedder,
multilingual (the Nepali path stays open), runs on GPU via
sentence-transformers CrossEncoder.

Wiring note: add to ChunkHit in retrieval.py:
    rerank_score: float | None = None
and include it in ChunkHit.trace() — reranker scores belong in
query_traces.retrieval alongside dense/lexical evidence.
"""

from __future__ import annotations

import logging
import time
import torch
device = "cuda" if torch.cuda.is_available() else "cpu"

log = logging.getLogger("query.rerank")

DEFAULT_MODEL = "BAAI/bge-reranker-v2-m3"


class Reranker:
    def __init__(self, model_name: str = DEFAULT_MODEL,
                 batch_size: int = 8):
        from sentence_transformers import CrossEncoder
        log.info("loading reranker %s", model_name)
        self.model = CrossEncoder(model_name,device=device)
        log.info("reranker on %s", device)
        self.batch_size = batch_size

    def rerank(self, query: str, hits: list, top_k: int = 8) -> list:
        """Score (query, chunk) pairs, attach rerank_score to each hit,
        return the top_k hits in reranked order.

        Hits keep their dense/lexical/RRF evidence — the trace shows the
        full journey of every candidate, including the ones reranking
        demoted.
        """
        if not hits:
            return []

        t0 = time.perf_counter()
        pairs = [(query, h.content) for h in hits]
        scores = self.model.predict(pairs, batch_size=self.batch_size)
        for h, s in zip(hits, scores):
            h.rerank_score = float(s)

        ranked = sorted(hits, key=lambda h: h.rerank_score, reverse=True)
        log.debug("reranked %d candidates in %d ms", len(hits),
                  round((time.perf_counter() - t0) * 1000))
        return ranked[:top_k]
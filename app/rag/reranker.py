"""Cross-encoder reranking for flight retrieval candidates.

Why cross-encoder reranking
---------------------------
Bi-encoder (dense vector) retrieval optimises for recall: it compares a single
query embedding against millions of pre-computed document embeddings in O(log n)
ANN time.  The tradeoff is that the query and document are encoded independently,
so the model cannot attend to token-level interactions between them during scoring.
Retrieval order within the top-k candidates is therefore often imprecise.

A cross-encoder concatenates query and document and scores the pair jointly,
recovering the fine-grained relevance signal the bi-encoder loses.  We retrieve
RERANK_FETCH_K candidates from Qdrant and let the cross-encoder produce a final
ranking of RERANK_RETURN_N — quality win at the cost of O(RERANK_FETCH_K)
cross-encoder inference calls per query.

Model selection rationale
--------------------------
The default model is BAAI/bge-reranker-base.  The previous default was
cross-encoder/ms-marco-MiniLM-L-6-v2, which was pre-trained on Bing web search
logs (MS MARCO passage retrieval task).  Two failure modes emerge on flight records:

1. Length bias: ms-marco models were trained on passages of ≥100 words.  The
   structured flight-record strings produced by flight_doc() are 25-40 words.
   ms-marco cross-encoders systematically underrank short documents relative to
   longer ones because the score distribution was calibrated on longer inputs.

2. Domain mismatch: web queries and their answer passages share vocabulary and
   style.  Flight queries ("morning Delta flight ATL→ORD") pair with structured
   records containing IATA codes, HHMM integers, and distances — a surface form
   the ms-marco models have not seen during fine-tuning.

BAAI/bge-reranker-base was trained on heterogeneous retrieval tasks drawn from
the BEIR benchmark and achieves average NDCG@10 ≈ 0.49 across 18 diverse corpora
(vs. ms-marco-MiniLM-L-6-v2 ≈ 0.42).  Its training distribution includes
entity-centric and structured queries, making it substantially more robust to
flight-record surface forms.

Model size comparison
---------------------
Model                               Size    NDCG@10 (BEIR avg)
cross-encoder/ms-marco-MiniLM-L-6-v2  90 MB   0.42  (former default)
BAAI/bge-reranker-base               278 MB   0.49  (current default)
BAAI/bge-reranker-large             1300 MB   0.52  (GPU recommended)

Run scripts/compare_rerankers.py against your annotated eval set to benchmark
these models on your own query distribution before committing to a change.

Runtime
-------
The CrossEncoder is loaded lazily on the first rerank call and held in a
module-level singleton.  sentence-transformers.CrossEncoder is not thread-safe
to construct concurrently; a double-checked asyncio.Lock serialises the one-time
initialisation.  Subsequent calls skip the lock entirely.

Inference runs in asyncio.to_thread (blocking CPU/IO) so it does not stall the
event loop.  On CPU, bge-reranker-base scores 20 candidates in ~200-400 ms.
"""
from __future__ import annotations

import asyncio
import logging

log = logging.getLogger(__name__)

_reranker = None
_reranker_lock = asyncio.Lock()


def _load_and_score(query: str, docs: list[str], model_name: str, top_n: int) -> list[int]:
    """Load the cross-encoder singleton (if not yet loaded) and score all pairs.

    Args:
        query: Original user query used as the left side of each (query, doc) pair.
        docs: Candidate document strings produced by flight_doc(), one per flight.
        model_name: HuggingFace model ID for the cross-encoder.
        top_n: Return at most this many indices.

    Returns:
        Sorted list of indices into docs (best first), length min(top_n, len(docs)).
    """
    global _reranker
    if _reranker is None:
        from sentence_transformers import CrossEncoder
        log.info("reranker.load model=%s", model_name)
        _reranker = CrossEncoder(model_name)

    pairs = [(query, d) for d in docs]
    scores = _reranker.predict(pairs)
    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    return ranked[:top_n]


async def rerank(query: str, flights: list[dict], top_n: int) -> list[dict]:
    """Rerank a candidate flight list with a cross-encoder and return the top results.

    Runs inference in a thread pool to avoid blocking the event loop.  The
    singleton is initialised before any scoring thread can access it, guarded by
    a double-checked asyncio.Lock on first call.

    Args:
        query: Original user query used as the left side of each pair.
        flights: Candidate flight dicts from Qdrant (typically RERANK_FETCH_K items).
        top_n: Number of results to return after reranking.

    Returns:
        Reranked subset of flights, length min(top_n, len(flights)).
    """
    if not flights:
        return flights

    from app.core.config import settings
    from app.rag.flight_doc import flight_doc

    docs = [flight_doc(f) for f in flights]

    if _reranker is None:
        async with _reranker_lock:
            if _reranker is None:
                indices = await asyncio.to_thread(
                    _load_and_score, query, docs, settings.RERANKER_MODEL, top_n
                )
                return [flights[i] for i in indices]

    indices = await asyncio.to_thread(
        _load_and_score, query, docs, settings.RERANKER_MODEL, top_n
    )
    return [flights[i] for i in indices]

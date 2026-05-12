"""RAG reranking — reorder retrieval candidates under a joint query+doc model.

Reranking sits between candidate generation (`retrieve(method="hybrid", ...)`)
and LLM consumption. Hybrid retrieval over a wide candidate pool (e.g. top-30)
gives high recall; rerank then provides precision by scoring each
``(query, document)`` pair jointly under a model that's purpose-built for
relevance ranking. This decouples the two trade-offs that bi-encoders /
keyword retrieval can't resolve simultaneously.

Two implementations are shipped:

- :class:`CrossEncoderReranker` — local BAAI ``bge-reranker-v2-m3`` (or any
  cross-encoder loadable via ``sentence-transformers``). Zero external API
  surface, deterministic, ~100-300ms latency on CPU at top-50. Default for
  Alpha Engine consumers per the no-new-vendor posture.
- :class:`LLMJudgeReranker` — Anthropic Haiku with a 1-5 relevance rubric.
  Higher latency + cost than cross-encoder; configurable opt-in for
  scenarios that need rerank criteria beyond pure semantic similarity
  ("rerank by recency-weighted relevance", "rerank by financial
  materiality").

Both implementations share the :class:`Reranker` protocol and the in-process
:class:`RerankCache` (LRU, keyed by ``sha256(query) + chunk_id``). Cache
lifetime is the process / Lambda container — no cross-run persistence,
because query embeddings drift with corpus updates and rerank scores are
cheap-to-recompute relative to the LLM call they enable.
"""

from __future__ import annotations

import hashlib
import logging
import os
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Callable, Protocol, runtime_checkable

from .retrieval import RetrievalResult

logger = logging.getLogger(__name__)


# Cap how many ``(query, doc)`` pairs the in-process cache retains so a
# long-running Lambda container doesn't grow unbounded. 1024 entries is
# ~8 queries × top-50 reranks × 2x slack — plenty of headroom for the
# 6-sector × ~25-ticker research run's qual-tool burst.
_DEFAULT_CACHE_MAXSIZE = 1024


# ── Cache ───────────────────────────────────────────────────────────────────


class RerankCache:
    """Process-local LRU cache for rerank scores keyed by ``(query, chunk_id)``.

    Keeps a tight cap on memory (``maxsize`` entries, eviction in
    insertion order) so a hot Lambda container that processes many
    distinct queries doesn't accumulate unbounded state. Lifetime is
    the container — no cross-invocation persistence (the
    ``RAG_RERANK_CACHE_TTL`` knob is intentionally absent because
    Lambda /tmp + the implied IO cost would exceed the cost of the
    rerank itself for typical query volumes).
    """

    def __init__(self, maxsize: int = _DEFAULT_CACHE_MAXSIZE) -> None:
        self._store: OrderedDict[str, float] = OrderedDict()
        self._maxsize = maxsize

    @staticmethod
    def make_key(query: str, chunk_id: str | None) -> str:
        # chunk_id can be None for results that didn't carry a primary key
        # back from the retriever (legacy ``vector_score-only`` paths); fall
        # back to hashing the content snippet plus the doc tuple so we
        # still get a stable key per ``(query, doc)`` pair.
        suffix = chunk_id if chunk_id is not None else "no_chunk_id"
        digest = hashlib.sha256(query.encode("utf-8")).hexdigest()[:16]
        return f"{digest}:{suffix}"

    def get(self, key: str) -> float | None:
        if key not in self._store:
            return None
        self._store.move_to_end(key)
        return self._store[key]

    def put(self, key: str, score: float) -> None:
        if key in self._store:
            self._store.move_to_end(key)
        self._store[key] = score
        if len(self._store) > self._maxsize:
            self._store.popitem(last=False)

    def __len__(self) -> int:
        return len(self._store)


# ── Reranker protocol ───────────────────────────────────────────────────────


@runtime_checkable
class Reranker(Protocol):
    """Score-and-reorder a candidate list under a joint query+doc model.

    Implementations may consult a cache, but the protocol surface is
    pure: take a query + candidate list, return the same candidates
    reordered (and optionally truncated to ``top_k``) with
    per-result ``rerank_score`` populated.
    """

    name: str

    def rerank(
        self,
        query: str,
        candidates: list[RetrievalResult],
        top_k: int,
    ) -> list[RetrievalResult]:
        ...


# ── Cross-encoder (local model) ─────────────────────────────────────────────


@dataclass
class CrossEncoderReranker:
    """Local cross-encoder reranker.

    Default model is BAAI ``bge-reranker-v2-m3``: a multilingual
    cross-encoder published 2024 at ~600MB on disk, ~100-300ms latency
    per query at top-50 on CPU. Any sentence-transformers
    :class:`CrossEncoder`-compatible model can be substituted via
    ``model_name``.

    The underlying ``sentence-transformers`` install is gated behind
    the ``[rerank]`` extra so callers that only use vector/hybrid
    retrieval don't pay the ~2GB torch + transformers + model-download
    install cost. Importing this module does NOT load the model;
    initialization happens lazily on the first :meth:`rerank` call so
    a non-rerank import path stays cheap.
    """

    model_name: str = "BAAI/bge-reranker-v2-m3"
    cache: RerankCache = field(default_factory=RerankCache)
    name: str = "cross_encoder"
    # When unset, defer model load until first rerank() call. Tests
    # patch this directly with a callable returning predict-able scores
    # to exercise the score-aware reorder path without paying the
    # ~600MB model download.
    _model: object | None = None

    def _ensure_model(self) -> object:
        if self._model is not None:
            return self._model
        try:
            # Imported lazily so a bare ``from alpha_engine_lib.rag import
            # retrieve`` stays cheap on consumers that don't rerank.
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise ImportError(
                "CrossEncoderReranker requires sentence-transformers. "
                "Install with: pip install 'alpha-engine-lib[rerank]'"
            ) from exc
        logger.info("Loading cross-encoder model: %s", self.model_name)
        self._model = CrossEncoder(self.model_name)
        return self._model

    def rerank(
        self,
        query: str,
        candidates: list[RetrievalResult],
        top_k: int,
    ) -> list[RetrievalResult]:
        if not candidates:
            return []

        uncached_pairs: list[tuple[int, str]] = []
        scores: list[float | None] = [None] * len(candidates)
        for idx, cand in enumerate(candidates):
            key = self.cache.make_key(query, cand.chunk_id)
            cached = self.cache.get(key)
            if cached is not None:
                scores[idx] = cached
            else:
                uncached_pairs.append((idx, cand.content))

        if uncached_pairs:
            model = self._ensure_model()
            pair_inputs = [[query, content] for _, content in uncached_pairs]
            # ``predict`` returns one logit per pair; higher = more relevant.
            # Type cast through ``list(map(float, ...))`` keeps tests
            # happy when a numpy array is returned by the real model and
            # when a plain list is returned by the test fake.
            raw = model.predict(pair_inputs)  # type: ignore[attr-defined]
            fresh_scores = list(map(float, raw))
            for (idx, _content), score in zip(uncached_pairs, fresh_scores):
                scores[idx] = score
                self.cache.put(
                    self.cache.make_key(query, candidates[idx].chunk_id),
                    score,
                )

        return _attach_and_sort(candidates, scores, self.name, top_k)


# ── LLM-as-judge ────────────────────────────────────────────────────────────


# Default rubric — kept terse to fit a Haiku context window comfortably
# at top-50 candidates and to leave room for the candidate text itself.
# Scores follow a 1-5 integer Likert that the model returns as plain
# JSON for deterministic parsing.
_DEFAULT_LLM_RUBRIC = (
    "Rate the relevance of the following document to the query on a "
    "1-5 scale where 1=irrelevant, 3=tangentially related, 5=directly "
    "answers the query. Respond with ONLY a single integer between 1 "
    "and 5."
)


@dataclass
class LLMJudgeReranker:
    """LLM-as-judge reranker — one Haiku call per (query, doc) pair.

    More expensive + slower than the cross-encoder (one LLM round-trip
    per candidate vs. one batched local-model inference for the whole
    set) but more flexible: the rubric can encode criteria beyond
    semantic similarity ("rerank by recency-weighted financial
    materiality"). Configure via :attr:`rubric` at construction.

    Default ``rubric`` is a strict 1-5 Likert; output is parsed as
    ``int(response.strip()[0])`` to tolerate the occasional Haiku
    leading whitespace or trailing punctuation. Parses that fail
    produce a neutral score of 3 + a warning log; the caller's batch
    still completes.

    The Anthropic client is injected so consumers can plug in a
    pre-configured ``ChatAnthropic`` (langchain) or
    ``anthropic.Anthropic`` instance. The protocol surface is just
    ``client.messages.create(...)`` for the raw SDK shape.
    """

    client: object
    model: str = "claude-haiku-4-5-20251001"
    rubric: str = _DEFAULT_LLM_RUBRIC
    cache: RerankCache = field(default_factory=RerankCache)
    name: str = "llm_judge"

    def rerank(
        self,
        query: str,
        candidates: list[RetrievalResult],
        top_k: int,
    ) -> list[RetrievalResult]:
        if not candidates:
            return []

        scores: list[float | None] = [None] * len(candidates)
        for idx, cand in enumerate(candidates):
            key = self.cache.make_key(query, cand.chunk_id)
            cached = self.cache.get(key)
            if cached is not None:
                scores[idx] = cached
                continue
            score = self._score_one(query, cand.content)
            scores[idx] = score
            self.cache.put(key, score)

        return _attach_and_sort(candidates, scores, self.name, top_k)

    def _score_one(self, query: str, content: str) -> float:
        # Truncate the candidate text so a top-50 sweep at ~3K tokens per
        # candidate doesn't push the prompt past Haiku's window.
        snippet = content[:4000]
        prompt = (
            f"{self.rubric}\n\n"
            f"Query: {query}\n\n"
            f"Document:\n{snippet}\n\n"
            f"Score (1-5):"
        )
        try:
            response = self.client.messages.create(  # type: ignore[attr-defined]
                model=self.model,
                max_tokens=8,
                messages=[{"role": "user", "content": prompt}],
            )
            # Anthropic SDK response shape: response.content is a list of
            # content blocks; the first text block holds the integer.
            text_block = response.content[0]
            raw = getattr(text_block, "text", str(text_block)).strip()
            return float(int(raw[0]))
        except (ValueError, IndexError, AttributeError) as exc:
            logger.warning(
                "LLMJudgeReranker parse-fail (returning neutral 3): %s — raw=%r",
                exc, locals().get("raw", "<no response>"),
            )
            return 3.0


# ── Helpers ─────────────────────────────────────────────────────────────────


def _attach_and_sort(
    candidates: list[RetrievalResult],
    scores: list[float | None],
    method_name: str,
    top_k: int,
) -> list[RetrievalResult]:
    """Stamp ``rerank_score`` + ``rerank_method`` on each result and sort.

    ``RetrievalResult`` is a dataclass — set the fields directly. If the
    score list contains ``None`` for any candidate (shouldn't happen
    under correct caller flow, but defensive), those candidates sort to
    the tail so we don't drop them silently.
    """
    paired = list(zip(candidates, scores))
    paired.sort(key=lambda x: (x[1] is None, -(x[1] or 0.0)))
    out: list[RetrievalResult] = []
    for cand, score in paired[:top_k]:
        cand.rerank_score = score  # type: ignore[attr-defined]
        cand.rerank_method = method_name  # type: ignore[attr-defined]
        out.append(cand)
    return out


# ── Factory for the retrieve() integration ──────────────────────────────────


# Module-level registry of named reranker instances. Lazily populated
# the first time :func:`get_reranker` resolves a given name, then
# memoized so subsequent retrieve(rerank="cross_encoder", ...) calls
# share the same cache + model handle within the Lambda container.
_RERANKER_REGISTRY: dict[str, Reranker] = {}


# Factory hook used by :func:`get_reranker` for the ``"llm_judge"``
# case — exposed at module scope so tests can patch it without
# importing the anthropic SDK. Default constructs an Anthropic client
# from the environment, matching the pattern used elsewhere in
# alpha-engine-research.
def _default_llm_judge_factory() -> Reranker:
    try:
        from anthropic import Anthropic  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ImportError(
            "LLMJudgeReranker requires the anthropic SDK. "
            "Install via: pip install anthropic"
        ) from exc
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError(
            "LLMJudgeReranker needs ANTHROPIC_API_KEY in the environment."
        )
    return LLMJudgeReranker(client=Anthropic(api_key=api_key))


_LLM_JUDGE_FACTORY: Callable[[], Reranker] = _default_llm_judge_factory


def get_reranker(name: str) -> Reranker:
    """Resolve a named reranker, constructing + caching on first use.

    Supported names: ``"cross_encoder"`` (default — local BAAI),
    ``"llm_judge"`` (Anthropic Haiku via the ``anthropic`` SDK).
    Tests register fakes by writing directly to
    :data:`_RERANKER_REGISTRY` before the ``retrieve(rerank=...)`` call.
    """
    if name in _RERANKER_REGISTRY:
        return _RERANKER_REGISTRY[name]
    if name == "cross_encoder":
        instance: Reranker = CrossEncoderReranker()
    elif name == "llm_judge":
        instance = _LLM_JUDGE_FACTORY()
    else:
        raise ValueError(
            f"Unknown reranker {name!r}; supported: 'cross_encoder', 'llm_judge'"
        )
    _RERANKER_REGISTRY[name] = instance
    return instance

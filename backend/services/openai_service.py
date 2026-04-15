"""
OpenAI service.
Handles embedding generation, chat completions, and LLM-based relevance selection.
"""

import asyncio
import json
import logging
import re
from typing import Any, AsyncGenerator

from openai import AsyncOpenAI, OpenAI

from backend.config import Settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Deterministic fallback used when all reranker attempts fail
# ---------------------------------------------------------------------------

# Matches numeric values followed by common technical units.
# Used to identify chunks that contain concrete measured facts.
_NUMERIC_UNIT_RE = re.compile(
    r"\d+(?:\.\d+)?\s*"
    r"(?:%"
    r"|mm\b|cm\b|km\b"
    r"|kPa\b|MPa\b|Pa\b"
    r"|kg\b|mg\b"
    r"|hours?\b|hrs?\b|minutes?\b|mins?\b|seconds?\b|secs?\b"
    r"|[°]\s*[CcFf]"
    r")",
    re.IGNORECASE,
)


def _keyword_match_score(query: str, text: str) -> float:
    """
    Return a 0.0–1.0 keyword relevance score for a single chunk.

    Scoring:
      base  — fraction of meaningful query words (>= 3 chars) found as whole
              words in the chunk text.
      bonus — +0.25 if the chunk contains any numeric+unit pattern
              (%, mm, kPa, hours, °C …), because fact-based questions almost
              always need a chunk that carries a concrete measured value.

    The bonus is capped so the total never exceeds 1.0.
    The calculation is fully deterministic: same query + same text → same score.
    """
    if not text or not query:
        return 0.0

    text_lower  = text.lower()
    query_lower = query.lower()

    # Tokenize query into meaningful words (skip stop-word-length tokens)
    query_words = [w for w in re.findall(r"\b\w+\b", query_lower) if len(w) >= 3]
    if not query_words:
        return 0.0

    # Tokenize chunk into a word set for O(1) whole-word lookup
    chunk_word_set = set(re.findall(r"\b\w+\b", text_lower))

    hit_count  = sum(1 for w in query_words if w in chunk_word_set)
    base_score = hit_count / len(query_words)

    # Boost chunks that contain a numeric measurement — these are the chunks
    # most likely to hold exact answer values like "75%", "2 hours", "120°C".
    numeric_bonus = 0.25 if _NUMERIC_UNIT_RE.search(text) else 0.0

    return min(base_score + numeric_bonus, 1.0)


def _score_sorted_fallback(chunks: list[dict], top_k: int, query: str = "") -> list[dict]:
    """
    Keyword-aware deterministic fallback ranking.

    Sort priority (all keys are deterministic for the same input):
      1. Keyword match score (desc) — chunks whose text overlaps most with
         the query, boosted further if they contain numeric+unit values.
      2. Vector/RRF relevance score (desc) — original retrieval confidence.
      3. Document name (asc) — alphabetical tie-break across documents.
      4. Chunk index (asc) — earlier position in the same document wins.

    This ensures that a chunk containing "75%" will rank above a generic
    chunk even when the LLM reranker is unavailable.
    """
    sorted_chunks = sorted(
        chunks,
        key=lambda c: (
            -_keyword_match_score(query, c.get("text_chunk", "")),
            -(c.get("rrf_score") or c.get("score", 0.0)),
            c.get("document_name", ""),
            c.get("chunk_index", 0),
        ),
    )
    return sorted_chunks[:top_k]


class OpenAIService:
    """Wrapper around the OpenAI API for embeddings, completions, and relevance selection."""

    def __init__(self, config: Settings):
        if not config.openai_api_key or not config.openai_api_key.strip():
            raise ValueError(
                "OPENAI_API_KEY is not set. All OpenAI API calls will fail. "
                "Set OPENAI_API_KEY in your environment before starting the server."
            )
        self.config = config
        # max_retries=3: SDK automatically retries rate-limit (429) and server
        # errors (500/503) with exponential backoff before raising an exception.
        self._async_client = AsyncOpenAI(api_key=config.openai_api_key, max_retries=3)
        self._sync_client = OpenAI(api_key=config.openai_api_key, max_retries=3)

    # ------------------------------------------------------------------
    # Embeddings
    # ------------------------------------------------------------------

    async def generate_embedding(self, text: str) -> list[float]:
        """Generate a single embedding vector for the given text."""
        text = text.replace("\n", " ").strip()
        response = await self._async_client.embeddings.create(
            input=[text],
            model=self.config.openai_embedding_model,
        )
        return response.data[0].embedding

    async def generate_embeddings(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a list of texts, batched in groups of 100."""
        all_embeddings: list[list[float]] = []
        batch_size = 100
        for i in range(0, len(texts), batch_size):
            batch = [t.replace("\n", " ").strip() for t in texts[i: i + batch_size]]
            response = await self._async_client.embeddings.create(
                input=batch,
                model=self.config.openai_embedding_model,
            )
            sorted_data = sorted(response.data, key=lambda x: x.index)
            all_embeddings.extend([item.embedding for item in sorted_data])
            logger.debug("Generated embeddings for batch %d-%d", i, i + len(batch) - 1)
        return all_embeddings

    def generate_embedding_sync(self, text: str) -> list[float]:
        return asyncio.get_event_loop().run_until_complete(self.generate_embedding(text))

    def generate_embeddings_sync(self, texts: list[str]) -> list[list[float]]:
        return asyncio.get_event_loop().run_until_complete(self.generate_embeddings(texts))

    # ------------------------------------------------------------------
    # Chat Completions
    # ------------------------------------------------------------------

    async def chat_completion(
        self,
        messages: list[dict[str, Any]],
        max_tokens: int | None = None,
    ) -> str:
        """
        Generate a chat completion.

        temperature=0.0 + seed=openai_seed maximises reproducibility:
        same prompt → same output within a model version.
        """
        if max_tokens is None:
            max_tokens = self.config.max_tokens
        response = await self._async_client.chat.completions.create(
            model=self.config.openai_chat_model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=0.0,
            seed=self.config.openai_seed,
        )
        return response.choices[0].message.content or ""

    async def chat_completion_stream(
        self,
        messages: list[dict[str, Any]],
        max_tokens: int | None = None,
    ) -> AsyncGenerator[str, None]:
        """Stream a chat completion response."""
        if max_tokens is None:
            max_tokens = self.config.max_tokens
        stream = await self._async_client.chat.completions.create(
            model=self.config.openai_chat_model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=0.0,
            seed=self.config.openai_seed,
            stream=True,
        )
        async for chunk in stream:
            content = chunk.choices[0].delta.content if chunk.choices else None
            if content:
                yield content

    # ------------------------------------------------------------------
    # LLM-based relevance selection (reranker)
    # ------------------------------------------------------------------

    async def rerank_chunks(
        self,
        query: str,
        chunks: list[dict],
        top_k: int,
    ) -> list[dict]:
        """
        LLM-based relevance selection: narrows the candidate chunk pool to top_k.

        This is NOT a trained cross-encoder reranker.  A fast LLM reads all
        candidate chunks in a single prompt and selects the most relevant ones.

        Reliability guarantees:
          - Runs with a per-attempt timeout (rerank_timeout_seconds).
          - Retries once on timeout or API error (rerank_max_retries=1).
          - Falls back to a deterministic score-sorted order if all attempts fail.
          - The fallback is stable: same input always produces the same output.
        """
        if len(chunks) <= top_k:
            return chunks

        chunk_previews = "\n\n".join(
            f"[{i + 1}] {c['text_chunk'][:500].strip()}"
            for i, c in enumerate(chunks)
        )

        rerank_prompt = (
            f"You are a relevance judge for document retrieval.\n\n"
            f"User question: {query}\n\n"
            f"Below are {len(chunks)} document excerpts (numbered):\n\n"
            f"{chunk_previews}\n\n"
            f"Task: Return the numbers of the {top_k} most relevant excerpts, "
            f"ordered from most to least relevant. "
            f"Prioritise excerpts that directly answer the question with specific "
            f"facts, values, numbers, or explanations.\n"
            f"Return ONLY a JSON array of integers, e.g. [3, 1, 7]. No explanation."
        )

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a precise relevance judge. "
                    "Output only a valid JSON array of integers. No other text."
                ),
            },
            {"role": "user", "content": rerank_prompt},
        ]

        max_attempts = self.config.rerank_max_retries + 1
        for attempt in range(1, max_attempts + 1):
            try:
                result = await self._call_reranker_once(messages, chunks, top_k)
                if not result:
                    raise ValueError("Reranker returned an empty result set.")
                logger.info(
                    "Relevance selection succeeded (attempt %d/%d).",
                    attempt, max_attempts,
                )
                return result
            except asyncio.TimeoutError:
                logger.warning(
                    "Relevance selection timed out after %ds (attempt %d/%d).",
                    self.config.rerank_timeout_seconds, attempt, max_attempts,
                )
            except Exception as exc:
                logger.warning(
                    "Relevance selection failed (attempt %d/%d): %s",
                    attempt, max_attempts, exc,
                )

        logger.warning(
            "All %d relevance selection attempt(s) failed. "
            "Using keyword-aware deterministic fallback.",
            max_attempts,
        )
        try:
            return _score_sorted_fallback(chunks, top_k, query=query)
        except Exception as exc:  # pragma: no cover
            logger.error("Fallback ranking failed (%s); returning raw slice.", exc)
            return chunks[:top_k]

    async def _call_reranker_once(
        self,
        messages: list[dict[str, Any]],
        chunks: list[dict],
        top_k: int,
    ) -> list[dict]:
        """
        Single reranker API call with timeout enforcement.
        Raises asyncio.TimeoutError, ValueError, or any OpenAI API exception.
        """
        raw = await asyncio.wait_for(
            self._async_client.chat.completions.create(
                model=self.config.openai_rerank_model,
                messages=messages,
                max_tokens=120,
                temperature=0.0,   # Must be 0.0 for consistent selection
            ),
            timeout=self.config.rerank_timeout_seconds,
        )
        raw_text = raw.choices[0].message.content or ""

        match = re.search(r"\[[\d,\s]+\]", raw_text)
        if not match:
            raise ValueError(f"Non-parseable reranker output: {raw_text[:80]!r}")

        indices: list[int] = json.loads(match.group())

        seen: set[int] = set()
        reranked: list[dict] = []
        for idx in indices:
            if isinstance(idx, int) and 1 <= idx <= len(chunks):
                i = idx - 1
                if i not in seen:
                    seen.add(i)
                    reranked.append(chunks[i])
                    if len(reranked) == top_k:
                        break

        # Pad to top_k if the reranker returned fewer indices than requested
        if len(reranked) < top_k:
            for i, chunk in enumerate(chunks):
                if i not in seen:
                    reranked.append(chunk)
                if len(reranked) == top_k:
                    break

        logger.debug("Reranker selected indices %s from %d candidates.", indices, len(chunks))
        return reranked

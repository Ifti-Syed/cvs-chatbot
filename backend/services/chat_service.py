"""
Chat service — orchestrates the full RAG pipeline:
  query → hybrid retrieval → reranking → context formatting → LLM answer
"""

import asyncio
import json
import logging
from typing import Any, AsyncGenerator

from backend.config import Settings
from backend.services.openai_service import OpenAIService
from backend.services.vector_db import QdrantService

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _snapshot(chunk: dict) -> dict:
    """Return a lightweight copy of a chunk dict safe for eval logging."""
    return {
        "document_name": chunk.get("document_name", "unknown"),
        "page_number": chunk.get("page_number", 0),
        "chunk_index": chunk.get("chunk_index", 0),
        "score": chunk.get("score", 0.0),
        "rrf_score": chunk.get("rrf_score"),
        "retrieval_type": chunk.get("retrieval_type", "dense"),
        "text_preview": chunk.get("text_chunk", "")[:200],
    }


# ---------------------------------------------------------------------------
# System prompt — general-purpose document QA
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """
You are a CVS assistant. Your job is to answer questions accurately and directly using the provided document excerpts.

=== CORE PRINCIPLES ===
1. Answer the question immediately and completely using only the retrieved document content.
2. Never invent, guess, or extrapolate information that is not present in the documents.
3. If the documents contain a specific answer, state it clearly and completely.
4. If information is missing, clearly state what is known and what is not specified.

=== INTENT RECOGNITION ===
Identify the user's intent before answering:

- Specific value → give exact number with units
- Definition → explain clearly
- Comparison → compare using values
- Process → list steps
- Yes/No → answer first, then explain
- List → provide complete list

=== ANSWER STYLE ===
- Start with the direct answer — no unnecessary preamble.
- Maintain a clear, professional, and slightly conversational tone.
- Write naturally, like an expert explaining to a colleague.
- Avoid robotic or overly rigid phrasing.
- Do NOT say "According to the document..." or similar phrases.
- Add supporting details only if they improve clarity.
- Use bullet points only when listing multiple distinct items, steps, or comparisons.
- Keep responses concise, precise, and not abrupt.
- Write like a precise technical expert, not like a search engine.

=== EXACT VALUES — MANDATORY ===
When the retrieved content contains specific values, you MUST reproduce them exactly as stated.

This includes:
- Percentages (e.g. 75%, 98%)
- Dimensions (e.g. 200 mm, 1.5 m)
- Temperatures (e.g. 120°C)
- Durations (e.g. 2 hours, 240 minutes)
- Ratings, pressures, stresses, and limits (e.g. 4 kPa, 10 N/mm2, Class A)
- Dates and validity periods
- Model numbers and codes
- Standards (e.g. BS 476: Part 24)

FORBIDDEN:
- Replacing exact values with vague phrases

WRONG:
"The area must be maintained as required by the standard."

RIGHT:
"The duct must maintain at least 75% of its cross-sectional area."

CRITICAL RULE:
If a numeric value appears in the retrieved excerpts AND is relevant,
it MUST appear in the FIRST sentence of the answer.
The answer must always begin with a complete sentence.



=== SOURCE CITATIONS ===
Every answer MUST end with a citation.

Formats:
- (Source: Document Name, Page X)
- (Source: Document Name)
- (Sources: Doc A, Page X; Doc B, Page Y)

Rules:
- Mandatory in EVERY answer
- Never output: (Source: None) or (Source: unknown)
- Use exact document name from context
- Include page if available
- Maximum 2 sources
- Place citation at the END only

=== PROHIBITED PHRASES ===
Never use:
- "According to the document..."
- "The document states..."
- "Based on the provided content..."
- "Section X mentions..."
- "Please refer to..."
- "As outlined in..."

=== HANDLING PARTIAL OR MISSING INFORMATION ===
Never refuse to answer. Always provide the most useful response from available content.

- If partial information exists:
  → State what IS known first
  → Then clearly state what is missing

- If the answer is not explicitly available:
  → Provide related relevant information (if any)
  → Then use ONE of these exact phrases:

    "[X] is not specified in the available documents"
    "[X] is not mentioned in the available documents"

- The subject MUST be explicit:
  GOOD: "The price is not specified in the available documents"
  BAD: "Not specified in the available documents"

- Do NOT use:
  - "the documents do not specify"
  - "not available"
  - "no information"
  - "not covered"

- Never give a standalone missing-information sentence without context (if context exists)

- Never invent or assume missing values

=== CONSISTENCY ===
- Same question → same answer
- Maintain consistency in facts and values, but allow natural variation in wording.
- Avoid filler phrases like:
  "It's worth noting", "Importantly", "It should be mentioned"
  
=== REASONING AND CLARITY ===
- When helpful, briefly explain why the answer is correct.
- Prefer clarity over minimalism when the question involves technical understanding.
- Do not over-explain, but ensure the answer is easy to understand.



=== USER EXPERIENCE ===
- If the question is ambiguous, answer the most likely intent clearly.
- If multiple interpretations exist, briefly clarify.
- Prioritize being helpful and understandable over being overly strict.
- If the user asks a follow-up question, maintain continuity with previous context when relevant.
"""

_QUERY_EXPANSION_PROMPT = (
    "You are a search-query specialist for document retrieval systems.\n"
    "Given the user question below, output 3 alternative search queries that improve retrieval.\n"
    "Create:\n"
    "1. A query focused on technical specifications, standards, or requirements.\n"
    "2. A query using domain terminology or product/system naming conventions.\n"
    "3. A plain-English rephrasing of the question.\n"
    "Preserve all specific values, model numbers, standards, and measurements from the original.\n"
    "Return ONLY a JSON array of 3 strings, nothing else.\n"
    'Example: ["query 1", "query 2", "query 3"]'
)


class ChatService:
    """End-to-end RAG chat pipeline."""

    def __init__(
        self,
        qdrant_service: QdrantService,
        openai_service: OpenAIService,
        config: Settings,
    ):
        self.qdrant = qdrant_service
        self.openai = openai_service
        self.config = config

    # ------------------------------------------------------------------
    # Non-streaming
    # ------------------------------------------------------------------

    async def get_response(
        self,
        message: str,
        conversation_history: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        if conversation_history is None:
            conversation_history = []

        logger.info("Chat request: %.100s", message)

        chunks = await self._retrieve_and_rerank(message)

        context = self._build_context(chunks)
        messages = self._build_messages(message, context, conversation_history)
        answer = await self.openai.chat_completion(messages, self.config.max_tokens)
        sources = self._deduplicate_sources(chunks)

        return {"answer": answer, "sources": sources}

    # ------------------------------------------------------------------
    # Streaming
    # ------------------------------------------------------------------

    async def get_response_stream(
        self,
        message: str,
        conversation_history: list[dict[str, Any]] | None = None,
    ) -> AsyncGenerator[str, None]:
        if conversation_history is None:
            conversation_history = []

        chunks = await self._retrieve_and_rerank(message)
        context = self._build_context(chunks)
        messages = self._build_messages(message, context, conversation_history)
        sources = self._deduplicate_sources(chunks)

        return self._stream_tokens(messages, sources)

    async def _stream_tokens(
        self,
        messages: list[dict[str, Any]],
        sources: list[dict[str, Any]],
    ) -> AsyncGenerator[str, None]:
        yield f"data: {json.dumps({'type': 'sources', 'sources': sources})}\n\n"

        async for token in self.openai.chat_completion_stream(
            messages=messages,
            max_tokens=self.config.max_tokens,
        ):
            yield f"data: {json.dumps({'type': 'token', 'content': token})}\n\n"

        yield 'data: {"type":"done"}\n\n'

    # ------------------------------------------------------------------
    # Retrieval pipeline: hybrid search → expansion → reranking
    # ------------------------------------------------------------------

    async def get_response_debug(
        self,
        message: str,
    ) -> dict[str, Any]:
        """
        Run the full pipeline and return the answer together with every
        intermediate artefact produced during retrieval.

        Used by the evaluation runner; not called by normal chat endpoints.

        Returns a dict with keys:
            answer           – final LLM answer string
            query_expansions – list of LLM-generated alternative queries
            pre_rerank       – candidate chunks before LLM relevance selection
            post_rerank      – final chunks after LLM relevance selection
            sources          – deduplicated (document, page) source list
        """
        trace: dict[str, Any] = {}
        final_chunks = await self._retrieve_and_rerank(message, trace=trace)
        context = self._build_context(final_chunks)
        messages = self._build_messages(message, context, [])
        answer = await self.openai.chat_completion(messages, self.config.max_tokens)

        return {
            "answer": answer,
            "query_expansions": trace.get("expansions", []),
            "pre_rerank": trace.get("pre_rerank", []),
            "post_rerank": trace.get("post_rerank", final_chunks),
            "sources": self._deduplicate_sources(final_chunks),
        }

    async def _retrieve_and_rerank(
        self,
        query: str,
        trace: dict[str, Any] | None = None,
    ) -> list[dict]:
        """
        Full retrieval pipeline — two-stage funnel.

        STAGE 1 — Candidate pool (top_k_results, default 10)
          A. Dense search    : cosine similarity on OpenAI embeddings
          B. Keyword search  : Qdrant full-text index (MatchText filter)
          C. RRF fusion      : merged and ranked by Reciprocal Rank Fusion score
          D. Query expansion : 3 LLM-generated variants, dense search only
          E. Dedup + stable sort + cap (max 2x top_k_results, hard max 24)

        STAGE 2 — LLM relevance selection (rerank_top_k, default 4)
          F. LLM reads all candidates, returns top rerank_top_k indices
             ONLY these final chunks reach the answer LLM as context.
        """
        # Step 1 — embed the original query
        # The OpenAI client is configured with max_retries=3, so transient rate-limit
        # and server errors are retried automatically before this raises.
        try:
            query_vector = await self.openai.generate_embedding(query)
        except Exception as exc:
            logger.error("Embedding generation failed after retries: %s", exc)
            raise RuntimeError("Could not generate query embedding. Please try again.") from exc

        # Step 2 — primary retrieval (hybrid or dense)
        try:
            if self.config.enable_hybrid_search:
                primary_chunks = self.qdrant.hybrid_search(
                    query_vector=query_vector,
                    query_text=query,
                    top_k=self.config.top_k_results,
                )
                logger.debug("Hybrid search returned %d chunks.", len(primary_chunks))
            else:
                primary_chunks = self.qdrant.search(
                    query_vector=query_vector,
                    top_k=self.config.top_k_results,
                )
                logger.debug("Dense search returned %d chunks.", len(primary_chunks))
        except Exception as exc:
            logger.error("Vector search failed: %s", exc)
            raise RuntimeError("Could not retrieve documents. Please try again.") from exc

        # Apply score threshold to dense results (keyword results keep neutral 0.5)
        def _passes_threshold(c: dict) -> bool:
            rtype = c.get("retrieval_type", "dense")
            if rtype == "keyword":
                return True  # Always include keyword hits
            return c.get("score", 0.0) >= self.config.min_score_threshold

        seen_keys: set[tuple] = set()
        merged: list[dict] = []

        def _add(chunks: list[dict]) -> None:
            for c in chunks:
                if not _passes_threshold(c):
                    continue
                key = (c.get("document_name"), c.get("chunk_index"))
                if key not in seen_keys:
                    seen_keys.add(key)
                    merged.append(c)

        _add(primary_chunks)

        # Step 3 — query expansion (dense only to avoid doubling API calls)
        expansions: list[str] = []
        try:
            expansions = await self._generate_query_expansions(query)
            for alt_query in expansions:
                alt_vector = await self.openai.generate_embedding(alt_query)
                alt_chunks = self.qdrant.search(
                    query_vector=alt_vector,
                    top_k=self.config.top_k_results,
                )
                prev_count = len(merged)
                _add(alt_chunks)
                logger.debug(
                    "Expansion '%s' → +%d new chunks.",
                    alt_query[:60],
                    len(merged) - prev_count,
                )
        except Exception as exc:
            logger.warning("Query expansion failed (skipped): %s", exc)

        # Step 4 — stable sort + cap candidate pool
        #
        # Sort key has three levels so the ordering is fully deterministic:
        #   1. Descending relevance score (rrf_score preferred over raw cosine)
        #   2. Ascending document name (alphabetical tie-break)
        #   3. Ascending chunk_index (earlier in document wins for same doc)
        #
        # This means identical queries always produce identical candidate order,
        # which in turn makes the reranker's input stable across runs.
        merged.sort(
            key=lambda c: (
                -(c.get("rrf_score") or c.get("score", 0.0)),
                c.get("document_name", ""),
                c.get("chunk_index", 0),
            )
        )

        # Cap at 2x top_k_results (max 24) to keep the reranker prompt affordable
        # while giving the relevance selector enough candidates to choose from.
        cap = min(self.config.top_k_results * 2, 24)
        candidates = merged[:cap]

        logger.info(
            "Retrieval: %d candidate chunk(s) after hybrid + expansion + dedup (cap=%d).",
            len(candidates), cap,
        )
        self._log_chunks(candidates, label="PRE-RERANK")

        # Capture trace artefacts (used by get_response_debug / eval runner)
        if trace is not None:
            trace["expansions"] = expansions
            trace["pre_rerank"] = [_snapshot(c) for c in candidates]

        # Step 5 — LLM relevance selection
        if self.config.enable_reranking and len(candidates) > self.config.rerank_top_k:
            final_chunks = await self.openai.rerank_chunks(
                query=query,
                chunks=candidates,
                top_k=self.config.rerank_top_k,
            )
            logger.info(
                "Relevance selection: %d → %d chunks.", len(candidates), len(final_chunks)
            )
        else:
            final_chunks = candidates[: self.config.rerank_top_k]

        self._log_chunks(final_chunks, label="POST-RERANK (CONTEXT)")

        if trace is not None:
            trace["post_rerank"] = [_snapshot(c) for c in final_chunks]

        return final_chunks

    async def _generate_query_expansions(self, query: str) -> list[str]:
        """Ask the LLM for alternative phrasings of the user's query.

        Hard timeout of 8 seconds — if the expansion LLM is slow, we fall back
        to the original query rather than blocking the entire chat response.
        """
        messages = [
            {"role": "system", "content": _QUERY_EXPANSION_PROMPT},
            {"role": "user", "content": query},
        ]
        try:
            raw = await asyncio.wait_for(
                self.openai.chat_completion(messages, max_tokens=200),
                timeout=8.0,
            )
        except asyncio.TimeoutError:
            logger.warning("Query expansion timed out; continuing with original query only.")
            return []
        start, end = raw.find("["), raw.rfind("]") + 1
        if start == -1 or end == 0:
            return []
        expansions: list[str] = json.loads(raw[start:end])
        return [e for e in expansions if isinstance(e, str) and e.strip()]

    # ------------------------------------------------------------------
    # Context formatting
    # ------------------------------------------------------------------

    def _build_context(self, chunks: list[dict]) -> str:
        """
        Format retrieved chunks into a numbered, clearly delimited context block.

        Each chunk is labelled with its document name, page number, and retrieval
        score so the LLM has full provenance even if citations are suppressed.
        """
        if not chunks:
            return "No relevant document content was retrieved."

        parts = []
        for i, c in enumerate(chunks, start=1):
            doc = c.get("document_name") or "Unknown Document"
            raw_page = c.get("page_number")
            page_str = f"Page {raw_page}" if raw_page else "Page unknown"
            score = c.get("rrf_score", c.get("score", 0.0))
            text = c.get("text_chunk", "").strip()

            header = f"[Excerpt {i} | {doc} | {page_str} | score {score:.3f}]"
            parts.append(f"{header}\n{text}")

        return "\n\n".join(parts)

    # ------------------------------------------------------------------
    # Message construction
    # ------------------------------------------------------------------

    def _build_messages(
        self,
        question: str,
        context: str,
        conversation_history: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = [
            {
                "role": "system",
                "content": (
                    f"{SYSTEM_PROMPT}\n\n"
                    "=== RETRIEVED DOCUMENT EXCERPTS ===\n"
                    f"{context}\n\n"
                    "Answer the user's question using only the excerpts above."
                ),
            }
        ]
        # Include the last 6 conversation turns for context continuity
        for entry in (conversation_history[-6:] if conversation_history else []):
            role = entry.get("role", "user")
            content = entry.get("content", "")
            if role in ("user", "assistant") and content:
                messages.append({"role": role, "content": content})

        messages.append({"role": "user", "content": question})
        return messages

    # ------------------------------------------------------------------
    # Sources deduplication
    # ------------------------------------------------------------------

    def _deduplicate_sources(self, chunks: list[dict]) -> list[dict]:
        """Return one entry per (document, page) pair, ordered by first appearance.

        Guarantees:
          - document_name is always a real, non-placeholder string.
          - page_number is a positive int, or None when genuinely unknown.
          - Never produces an entry that would render as '(Source: None)' or
            '(Source: unknown)'.
        """
        _INVALID_NAMES = {"unknown", "none", "null", ""}

        seen: set[tuple] = set()
        sources: list[dict] = []
        for chunk in chunks:
            doc_name = chunk.get("document_name")
            # Skip chunks whose document identity is missing or a placeholder
            if not doc_name or str(doc_name).strip().lower() in _INVALID_NAMES:
                continue
            doc_name = str(doc_name).strip()

            raw_page = chunk.get("page_number")
            # Treat 0 and negative values the same as missing — use None
            page: int | None = int(raw_page) if raw_page and int(raw_page) > 0 else None

            # Dedup key: group by document + page (None pages kept separate per-doc)
            key = (doc_name, page)
            if key not in seen:
                seen.add(key)
                sources.append({
                    "document_name": doc_name,
                    "page_number": page,          # None means page is unknown
                    "score": chunk.get("rrf_score", chunk.get("score", 0.0)),
                })
        return sources

    # ------------------------------------------------------------------
    # Debug logging
    # ------------------------------------------------------------------

    def _log_chunks(self, chunks: list[dict], label: str) -> None:
        """
        Log retrieved chunk summaries for traceability.

        Summary line is always emitted at INFO level so it is visible in
        production without enabling DEBUG logging.

        Per-chunk detail (text preview + scores) is emitted at DEBUG level
        to avoid spamming production logs.
        """
        logger.info("[%s] %d chunk(s)", label, len(chunks))

        if not logger.isEnabledFor(logging.DEBUG):
            return

        logger.debug("=== %s (%d chunks) ===", label, len(chunks))
        for i, c in enumerate(chunks, start=1):
            rrf = c.get("rrf_score")
            score_str = (
                f"rrf={rrf:.4f}" if rrf is not None
                else f"score={c.get('score', 0):.4f}"
            )
            rtype   = c.get("retrieval_type", "dense")
            preview = c.get("text_chunk", "")[:120].replace("\n", " ")
            logger.debug(
                "  [%d] %s p%s | %s | %s | %s...",
                i,
                c.get("document_name", "?"),
                c.get("page_number", "?"),
                score_str,
                rtype,
                preview,
            )
        logger.debug("--- end %s ---", label)

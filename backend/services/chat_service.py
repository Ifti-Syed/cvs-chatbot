"""
Chat service — orchestrates the full RAG pipeline:
  query → classify → source-filtered hybrid retrieval → reranking → answer

Query classification drives retrieval:
  "table"   → filter source_type=json_table  (exact spec lookups)
  "image"   → filter source_type=image_caption (diagram / figure queries)
  "general" → search all source types
"""

import asyncio
import json
import logging
import re
from typing import Any, AsyncGenerator

from backend.config import Settings
from backend.services.openai_service import OpenAIService
from backend.services.vector_db import QdrantService

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Query classification
# ---------------------------------------------------------------------------

# Patterns that signal an HR / people-policy question
_HR_RE = re.compile(
    r'\b(hr\b|human\s+resources?|hrm\b|leave|annual\s+leave|sick\s+leave|'
    r'vacation|holiday|public\s+holiday|policy|policies|employee|employment|'
    r'salary|payroll|compensation|benefit|attendance|working\s+hours?|'
    r'overtime|disciplin|grievance|complaint|resignation|termination|'
    r'notice\s+period|probation|recruit|hiring|onboard|appraisal|'
    r'performance\s+review|code\s+of\s+conduct|staff|personnel|'
    r'entitled|entitlement|allowance|bonus|increment)\b',
    re.IGNORECASE,
)

# Patterns that signal the user wants a diagram or figure
_IMAGE_RE = re.compile(
    r'\b(show|display|see|view|look\s+at|figure\s*\d*|fig\.?\s*\d*|'
    r'diagram|drawing|image|picture|illustration|appendix\s*[a-z]?|'
    r'sketch|detail|layout)\b',
    re.IGNORECASE,
)

# Patterns that signal a structured-data / specification lookup
_TABLE_RE = re.compile(
    r'\b(thickness|gauge|joint|hanger|bearer|spacing|fastening|'
    r'reinforcement|stiffener|length|diameter|bore|'
    r'size|fire\s+rating|120\s*min|240\s*min|120min|240min|minutes?|'
    r'seam|rivet|weld|flange|coupling|bolt|channel|rsa|rsc|'
    r'insulate|insulation|uninsulated|what\s+(size|gauge|type))\b',
    re.IGNORECASE,
)

# Queries that mention CF761/fire-test compliance context must stay "general"
# so the supplemental pdf_text retrieval path is used, not json_table-only.
_CF761_COMPLIANCE_RE = re.compile(
    r'\b(cf761|fire\s+test|compliance|certificate|test\s+report|'
    r'bs476|maintain|cross.?section|smoke\s+outlet|resist\s+fire|'
    r'requirement|criteria|standard|regulation|approved)\b',
    re.IGNORECASE,
)

# Explicit "Figure N" reference — used for direct figure lookup
_FIGURE_REF_RE = re.compile(r'\bfigure\s*(\d+)\b', re.IGNORECASE)
_APPENDIX_RE   = re.compile(r'\bappend[ie]x\s*([a-z])\b', re.IGNORECASE)


def _classify_query(query: str) -> str:
    """Return 'hr', 'table', 'image', or 'general'."""
    if _HR_RE.search(query):
        return "hr"
    if _IMAGE_RE.search(query):
        return "image"
    # CF761 compliance/certificate questions must stay "general" so the
    # supplemental pdf_text retrieval path runs — even if they contain
    # table-like keywords (e.g. "CF761 specification fire rating").
    if _TABLE_RE.search(query) and not _CF761_COMPLIANCE_RE.search(query):
        return "table"
    return "general"


def _extract_figure_ref(query: str) -> str | None:
    """Return a normalised figure name if the query names one explicitly."""
    m = _FIGURE_REF_RE.search(query)
    if m:
        return f"Figure {m.group(1)}"
    m = _APPENDIX_RE.search(query)
    if m:
        return f"Appendix {m.group(1).upper()}"
    return None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _snapshot(chunk: dict) -> dict:
    return {
        "document_name": chunk.get("document_name", "unknown"),
        "page_number":   chunk.get("page_number", 0),
        "chunk_index":   chunk.get("chunk_index", 0),
        "score":         chunk.get("score", 0.0),
        "rrf_score":     chunk.get("rrf_score"),
        "retrieval_type":chunk.get("retrieval_type", "dense"),
        "source_type":   chunk.get("source_type", ""),
        "text_preview":  chunk.get("text_chunk", "")[:200],
    }


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """
You are CVS Assistant — you answer questions about SAFE4 fire-rated ductwork specifications (CF761 system) and CVS company HR policies.
Your answers must be grounded ENTIRELY in the retrieved data shown below.

=== CORE RULES ===
1. Use ONLY the retrieved specification excerpts. Never invent, guess, or extrapolate.
2. If the data contains the answer, state it completely and accurately.
3. If the data does not contain the answer, respond professionally and direct the user to the appropriate contact (see HANDLING MISSING INFORMATION below).

=== TABLE / SPECIFICATION QUERIES ===
When answering duct size, gauge, joint type, hanger, bearer, or installation queries:
- Reproduce EXACT values as they appear in the data — no rounding, no approximation.
- Always state: duct size range, system/product code, and insulation status.
- If multiple records match, list each relevant one.
- Required fields to always include (where present in the data):
    • Duct size range
    • System / product code (e.g. RGIL120, CGIL240, HGDS120)
    • Joint type
    • Thickness / gauge
    • Fastening method
    • Reinforcement (if applicable)
    • Stiffener type / spacing (if applicable)
    • Hanger size / bearer size / bearer spacing (for hanger tables)

=== IMAGE / DIAGRAM QUERIES ===
When the user asks to "show", "display", "see", or asks about a specific figure:
- The relevant diagram is automatically shown below your response when available.
- Describe what the figure shows based on the retrieved caption data.
- End with exactly: "The diagram is shown below."
- If NO diagram was retrieved: "I don't have a figure for [X] in the available specifications."
- NEVER say "images are not available in the text excerpts" — images are served separately.

=== EXACT VALUES — MANDATORY ===
Every numeric value, code, dimension, or rating present in the retrieved data
MUST appear verbatim in your answer.

Examples of exact values you MUST reproduce:
  - Dimensions:        0.80mm, 1220mm, 3000mm, 30x30x1.70mm
  - Hanger sizes:      M8, M10, M12
  - Bearer sizes:      Slotted Channel 40x40x2.50mm, RSA 50x50x5.0mm
  - Joint types:       Spigot Coupling, 25 mm deep flange, 30 mm deep flange
  - Fastenings:        4.8mm Rivet @200mm, M8 Fastening bolt @300mm
  - Fire ratings:      120 min, 240 min
  - Spacing:           Max 750 mm to joint or stiffener, 1500 mm

FORBIDDEN: replacing exact values with vague phrases.
WRONG:  "The hanger size is appropriate for this duct."
RIGHT:  "The hanger size is M8 with a maximum bearer spacing of 1500 mm."

=== INTENT RECOGNITION ===
- Specific value requested → lead with the exact value and its units
- Comparison (e.g. 120 vs 240 min) → compare side by side with exact values
- List / all options → list every matching record from the data
- Yes / No → answer first, then give the supporting specification

=== ANSWER STYLE ===
- Start with the direct answer. No preamble.
- Professional, clear, precise — like an engineer answering a colleague.
- Use bullet points when listing multiple specs or comparing items.
- Do NOT say "According to the data", "The specification states", or similar.

=== SOURCE CITATIONS ===
Every answer MUST end with a citation — even when the answer is "not specified".
Formats (choose the right one based on source type):
  (Source: Table A9, CF761)              ← for json_table records
  (Source: Table A17, Installation Guidelines)
  (Source: Figure 3, CF761)              ← for image_caption records
  (Source: CF761, Page 4)                ← for PDF text excerpts (use page number from header)
  (Source: CF761 specification)          ← fallback when page number is unavailable
Rules:
  - Use the table_number from Specification headers, figure_name from Image headers,
    or document name + page number from Excerpt headers.
  - Maximum 2 sources.
  - Place citation at the END only — never in the middle of the answer.
  - Never output: (Source: None) or (Source: unknown).
  - ALWAYS include a citation, even for "not specified" answers.

=== PROHIBITED PHRASES ===
Never use:
  - "According to the document…"
  - "The document states…"
  - "Based on the provided content…"
  - "As outlined in…"
  - "Please refer to…"
  - "images are not included in the excerpts"

=== HR POLICY QUERIES ===
When answering questions about HR policies (leave, attendance, employment, salary, conduct, etc.):
- Answer based ONLY on the retrieved HR policy excerpts.
- Quote specific entitlements, durations, or clause numbers exactly as stated in the document.
- Do NOT mix HR policy answers with ductwork specification content.
- End with a citation in this format: (Source: HR Policy, Page N)
- If the answer is not in the HR policy excerpts, use the HR missing-information response below.

=== HANDLING MISSING INFORMATION ===
When the requested information is NOT present in the retrieved data, respond professionally
and direct the user to the appropriate contact. Never just say "not specified" alone.

For HR policy queries (leave, salary, attendance, conduct, employment terms, etc.):
  → "I don't have that information in the available HR policy documents. For further assistance,
     please contact the HR Manager directly.
     (Source: HR Policy)"

For ductwork / CF761 specification queries (dimensions, fire ratings, installation, compliance, etc.):
  → "That detail is not covered in the available CF761 documentation. For further technical guidance,
     please contact your Department Head or technical supervisor for approval.
     (Source: CF761 specification)"

Rules:
  - Be specific about what was not found (name the topic explicitly).
  - Always end with the appropriate citation even for not-found answers.
  - Keep the tone professional and helpful — never apologetic or vague.

GOOD: "The stiffener spacing for 120-min galvanised duct above 1250mm is not covered in the
       available CF761 documentation. Please contact your Department Head for further guidance.
       (Source: CF761 specification)"
BAD:  "Not specified in the available data."
"""

_QUERY_EXPANSION_PROMPT = (
    "You are a search-query specialist for a SAFE4 fire-rated ductwork specification system.\n"
    "Given the user question below, output 4 alternative search queries that improve retrieval.\n"
    "Create:\n"
    "1. A query using the exact product codes / system names (e.g. CGIL120, RGIL240, HGDS120).\n"
    "2. A query using technical specification terms (gauge, joint type, fastening, stiffener, hanger).\n"
    "3. A plain-English rephrasing.\n"
    "4. A fire test compliance rephrasing: use performance terms like 'maintain', 'must', "
    "'fire test', 'minimum', 'percentage', 'comply', 'smoke outlet', 'BS476' where relevant — "
    "targeting the CF761 fire test certificate rather than the product spec tables.\n"
    "Preserve all duct sizes, fire ratings, model codes, and measurements from the original.\n"
    "Return ONLY a JSON array of 4 strings, nothing else.\n"
    'Example: ["query 1", "query 2", "query 3", "query 4"]'
)


class ChatService:
    """End-to-end RAG chat pipeline with query-type-aware source filtering."""

    def __init__(
        self,
        qdrant_service: QdrantService,
        openai_service: OpenAIService,
        config: Settings,
    ):
        self.qdrant  = qdrant_service
        self.openai  = openai_service
        self.config  = config

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

        chunks, query_class = await self._retrieve_and_rerank(message)
        context = self._build_context(chunks)
        messages = self._build_messages(message, context, conversation_history, query_class)
        answer = await self.openai.chat_completion(messages, self.config.max_tokens)
        sources = self._deduplicate_sources(chunks, query_class)

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

        chunks, query_class = await self._retrieve_and_rerank(message)
        context = self._build_context(chunks)
        messages = self._build_messages(message, context, conversation_history, query_class)
        sources = self._deduplicate_sources(chunks, query_class)

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
    # Debug
    # ------------------------------------------------------------------

    async def get_response_debug(self, message: str) -> dict[str, Any]:
        trace: dict[str, Any] = {}
        final_chunks, query_class = await self._retrieve_and_rerank(message, trace=trace)
        context  = self._build_context(final_chunks)
        messages = self._build_messages(message, context, [], query_class)
        answer   = await self.openai.chat_completion(messages, self.config.max_tokens)

        return {
            "answer":          answer,
            "query_expansions":trace.get("expansions", []),
            "query_class":     trace.get("query_class", "general"),
            "pre_rerank":      trace.get("pre_rerank", []),
            "post_rerank":     trace.get("post_rerank", final_chunks),
            "sources":         self._deduplicate_sources(final_chunks),
        }

    # ------------------------------------------------------------------
    # Retrieval pipeline
    # ------------------------------------------------------------------

    async def _retrieve_and_rerank(
        self,
        query: str,
        trace: dict[str, Any] | None = None,
    ) -> list[dict]:
        """
        Two-stage funnel with query-type-aware source filtering.

        Stage 1 — Candidate pool
          A. Classify query → "table", "image", or "general"
          B. If image query and exact figure reference found → direct lookup
          C. Hybrid search (dense + keyword + RRF), optionally filtered by source_type
          D. Query expansion → dense-only search on 3 alternative queries
          E. Dedup + cap

        Stage 2 — LLM relevance selection
          F. LLM picks rerank_top_k best chunks from candidates
        """
        # ── Classify query ────────────────────────────────────────────────────
        query_class = _classify_query(query)
        figure_ref  = _extract_figure_ref(query) if query_class == "image" else None

        logger.info("Query class: %s | figure_ref: %s", query_class, figure_ref)
        if trace is not None:
            trace["query_class"] = query_class

        # ── Embed query ───────────────────────────────────────────────────────
        try:
            query_vector = await self.openai.generate_embedding(query)
        except Exception as exc:
            logger.error("Embedding failed: %s", exc)
            raise RuntimeError("Could not generate query embedding.") from exc

        # ── Direct figure lookup (exact name match) ───────────────────────────
        if figure_ref is not None:
            direct = self.qdrant.get_figure_by_name(figure_ref)
            if direct:
                logger.info("Direct figure lookup: '%s' → found.", figure_ref)
                if trace is not None:
                    trace["pre_rerank"] = [_snapshot(direct)]
                    trace["post_rerank"] = [_snapshot(direct)]
                return [direct], query_class
            logger.info("Direct figure lookup: '%s' → not found, falling back to search.", figure_ref)

        # ── Set source_type filter based on query class ───────────────────────
        # table → only JSON table records
        # image → only image captions
        # general → no filter (search everything)
        source_filter = None
        if query_class == "table":
            source_filter = "json_table"
        elif query_class == "image":
            source_filter = "image_caption"
        elif query_class == "hr":
            source_filter = "hr_policy"

        # ── Primary retrieval ─────────────────────────────────────────────────
        try:
            if self.config.enable_hybrid_search:
                primary_chunks = self.qdrant.hybrid_search(
                    query_vector=query_vector,
                    query_text=query,
                    top_k=self.config.top_k_results,
                    source_type_filter=source_filter,
                )
            else:
                primary_chunks = self.qdrant.search(
                    query_vector=query_vector,
                    top_k=self.config.top_k_results,
                    source_type_filter=source_filter,
                )
        except Exception as exc:
            logger.error("Vector search failed: %s", exc)
            raise RuntimeError("Could not retrieve specifications.") from exc

        # ── Score threshold (dense only) ──────────────────────────────────────
        def _passes(c: dict) -> bool:
            if c.get("retrieval_type") == "keyword":
                return True
            return c.get("score", 0.0) >= self.config.min_score_threshold

        seen: set[tuple] = set()
        merged: list[dict] = []

        def _add(chunks: list[dict]) -> None:
            for c in chunks:
                if not _passes(c):
                    continue
                key = (c.get("document_name"), c.get("chunk_index"))
                if key not in seen:
                    seen.add(key)
                    merged.append(c)

        _add(primary_chunks)

        # ── Supplemental pdf_text search (general queries only) ───────────────
        # CF761 compliance text chunks often score lower than spec tables in the
        # combined pool.  A separate pdf_text-only search ensures the top compliance
        # document chunks are always present in the candidate pool so the reranker
        # can pick them when the question is about test requirements or criteria.
        # Chunks are added WITHOUT the score threshold so low-scoring but critical
        # compliance text (e.g. the 75% cross-section requirement) is not filtered out.
        if query_class == "general":
            try:
                if self.config.enable_hybrid_search:
                    pdf_text_chunks = self.qdrant.hybrid_search(
                        query_vector=query_vector,
                        query_text=query,
                        top_k=25,
                        source_type_filter="pdf_text",
                    )
                else:
                    pdf_text_chunks = self.qdrant.search(
                        query_vector=query_vector,
                        top_k=25,
                        source_type_filter="pdf_text",
                    )
                for c in pdf_text_chunks:
                    key = (c.get("document_name"), c.get("chunk_index"))
                    if key not in seen:
                        seen.add(key)
                        merged.append(c)
            except Exception as exc:
                logger.warning("Supplemental pdf_text search skipped: %s", exc)

        # ── Query expansion (dense-only, same source filter) ──────────────────
        # Skipped for HR queries — the expansion prompt is SAFE4-specific and
        # would generate irrelevant fire-ductwork phrasings for HR questions.
        expansions: list[str] = []
        expansion_vectors: list[list[float]] = []
        if query_class != "hr":
            try:
                expansions = await self._generate_query_expansions(query)
                if expansions:
                    # Embed all expanded queries in parallel instead of sequentially —
                    # this cuts 4 serial API round-trips down to 1 concurrent batch.
                    expansion_vectors = list(
                        await asyncio.gather(
                            *[self.openai.generate_embedding(alt) for alt in expansions]
                        )
                    )
                    for alt_vec in expansion_vectors:
                        _add(self.qdrant.search(
                            query_vector=alt_vec,
                            top_k=self.config.top_k_results,
                            source_type_filter=source_filter,
                        ))
            except Exception as exc:
                logger.warning("Query expansion skipped: %s", exc)

        # ── Supplemental pdf_text expansion search (general queries) ──────────
        # Use each expansion vector (especially the compliance-angle 4th query)
        # to search pdf_text, so compliance chunks surfaced by the compliance-
        # phrased expansion land in the candidate pool without threshold filtering.
        if query_class == "general" and expansion_vectors:
            try:
                for alt_vec in expansion_vectors:
                    pdf_exp_chunks = self.qdrant.search(
                        query_vector=alt_vec,
                        top_k=6,
                        source_type_filter="pdf_text",
                    )
                    for c in pdf_exp_chunks:
                        key = (c.get("document_name"), c.get("chunk_index"))
                        if key not in seen:
                            seen.add(key)
                            merged.append(c)
            except Exception as exc:
                logger.warning("Supplemental pdf_text expansion search skipped: %s", exc)

        # ── Stable sort + cap ─────────────────────────────────────────────────
        merged.sort(
            key=lambda c: (
                -(c.get("rrf_score") or c.get("score", 0.0)),
                c.get("document_name", ""),
                c.get("chunk_index", 0),
            )
        )
        cap = min(self.config.top_k_results * 3, 40)
        candidates = merged[:cap]

        logger.info(
            "Retrieval: %d candidate(s) [class=%s, filter=%s, cap=%d]",
            len(candidates), query_class, source_filter or "all", cap,
        )
        self._log_chunks(candidates, label="PRE-RERANK")

        if trace is not None:
            trace["expansions"] = expansions
            trace["pre_rerank"] = [_snapshot(c) for c in candidates]

        # ── LLM relevance selection ───────────────────────────────────────────
        # Skip reranking for HR queries: the reranker prompt is CF761-biased
        # and would incorrectly deprioritise HR policy chunks, both producing
        # wrong answers and adding unnecessary latency.
        if (
            query_class != "hr"
            and self.config.enable_reranking
            and len(candidates) > self.config.rerank_top_k
        ):
            final_chunks = await self.openai.rerank_chunks(
                query=query,
                chunks=candidates,
                top_k=self.config.rerank_top_k,
            )
        else:
            final_chunks = candidates[: self.config.rerank_top_k]

        # ── Source diversity guarantee (general queries) ──────────────────────
        # Safety net: if the reranker dropped ALL pdf_text chunks despite the
        # improved prompt, force the highest-scoring one in so the answer is
        # grounded in the CF761 compliance document when relevant.
        if query_class == "general" and self.config.enable_reranking and final_chunks:
            pdf_in_final = any(c.get("source_type") == "pdf_text" for c in final_chunks)
            if not pdf_in_final:
                final_keys = {
                    (c.get("document_name"), c.get("chunk_index"))
                    for c in final_chunks
                }
                best_pdf = next(
                    (
                        c for c in candidates
                        if c.get("source_type") == "pdf_text"
                        and (c.get("document_name"), c.get("chunk_index")) not in final_keys
                    ),
                    None,
                )
                if best_pdf:
                    final_chunks[-1] = best_pdf
                    logger.info(
                        "Source diversity: swapped last chunk for pdf_text '%s' p%s.",
                        best_pdf.get("document_name"), best_pdf.get("page_number"),
                    )

        self._log_chunks(final_chunks, label="POST-RERANK")

        if trace is not None:
            trace["post_rerank"] = [_snapshot(c) for c in final_chunks]

        return final_chunks, query_class

    async def _generate_query_expansions(self, query: str) -> list[str]:
        messages = [
            {"role": "system", "content": _QUERY_EXPANSION_PROMPT},
            {"role": "user",   "content": query},
        ]
        try:
            raw = await asyncio.wait_for(
                self.openai.chat_completion(messages, max_tokens=200),
                timeout=8.0,
            )
        except asyncio.TimeoutError:
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
        if not chunks:
            return "No relevant specification data was retrieved."

        parts = []
        for i, c in enumerate(chunks, start=1):
            source_type = c.get("source_type", "")
            score = c.get("rrf_score", c.get("score", 0.0))
            text  = c.get("text_chunk", "").strip()

            if source_type == "json_table":
                doc        = c.get("document_name", "Unknown")
                table_num  = c.get("table_number", "")
                duct_size  = c.get("duct_size", "")
                system     = c.get("system", "")
                insulation = c.get("insulation", "")

                header_parts = [f"[Specification {i}"]
                if table_num:
                    header_parts.append(f"Table {table_num}")
                if doc:
                    header_parts.append(doc)
                if duct_size:
                    header_parts.append(f"Duct {duct_size}mm")
                if system:
                    header_parts.append(system)
                if insulation:
                    header_parts.append(insulation)
                header_parts.append(f"score {score:.3f}]")
                header = " | ".join(header_parts)

            elif source_type == "image_caption":
                figure_name = c.get("figure_name", c.get("document_name", "Figure"))
                header = f"[Image {i} | {figure_name} | score {score:.3f}]"

            else:
                doc  = c.get("document_name", "Unknown")
                page = c.get("page_number")
                page_str = f"Page {page}" if page else ""
                header = f"[Excerpt {i} | {doc}" + (f" | {page_str}" if page_str else "") + f" | score {score:.3f}]"

            parts.append(f"{header}\n{text}")

        return "\n\n".join(parts)

    # ------------------------------------------------------------------
    # Message construction
    # ------------------------------------------------------------------

    def _build_messages(
        self,
        question: str,
        context: str,
        history: list[dict[str, Any]],
        query_class: str = "general",
    ) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = [
            {
                "role": "system",
                "content": (
                    f"{SYSTEM_PROMPT}\n\n"
                    "=== RETRIEVED SPECIFICATION DATA ===\n"
                    f"{context}\n\n"
                    "Answer the CURRENT user question using ONLY the retrieved data above."
                ),
            }
        ]

        # HR and CF761/ductwork are completely separate knowledge domains.
        # Passing CF761 conversation history into an HR query (or vice-versa)
        # causes the LLM to distrust the freshly retrieved context and fall
        # back to "not found" responses.  Drop history entirely for HR queries
        # so the LLM answers solely from the HR policy excerpts above.
        if query_class != "hr":
            for entry in (history[-6:] if history else []):
                role    = entry.get("role", "user")
                content = entry.get("content", "")
                if role in ("user", "assistant") and content:
                    messages.append({"role": role, "content": content})

        messages.append({"role": "user", "content": question})
        return messages

    # ------------------------------------------------------------------
    # Sources deduplication
    # ------------------------------------------------------------------

    def _deduplicate_sources(self, chunks: list[dict], query_class: str = "general") -> list[dict]:
        _INVALID = {"unknown", "none", "null", ""}
        seen: set[tuple] = set()
        sources: list[dict] = []

        for chunk in chunks:
            doc_name = chunk.get("document_name")
            if not doc_name or str(doc_name).strip().lower() in _INVALID:
                continue
            doc_name = str(doc_name).strip()

            raw_page = chunk.get("page_number")
            page = int(raw_page) if raw_page and int(raw_page) > 0 else None

            key = (doc_name, page)
            if key not in seen:
                seen.add(key)
                # Only surface image_blob_names when the user explicitly asked
                # for a figure/diagram.  For all other query types the images
                # stay in the retrieval context (so the LLM can cite them) but
                # are never rendered in the chat — preventing spurious figure
                # thumbnails on non-image answers.
                show_images = query_class == "image"
                source = {
                    "document_name":    doc_name,
                    "page_number":      page,
                    "score":            chunk.get("rrf_score", chunk.get("score", 0.0)),
                    "image_blob_names": chunk.get("image_blob_names", []) if show_images else [],
                    "source_type":      chunk.get("source_type", ""),
                    "table_number":     chunk.get("table_number", ""),
                    "figure_name":      chunk.get("figure_name", ""),
                }
                sources.append(source)

        return sources

    # ------------------------------------------------------------------
    # Debug logging
    # ------------------------------------------------------------------

    def _log_chunks(self, chunks: list[dict], label: str) -> None:
        logger.info("[%s] %d chunk(s)", label, len(chunks))
        if not logger.isEnabledFor(logging.DEBUG):
            return

        for i, c in enumerate(chunks, start=1):
            rrf = c.get("rrf_score")
            score_str = f"rrf={rrf:.4f}" if rrf is not None else f"score={c.get('score',0):.4f}"
            preview = c.get("text_chunk", "")[:100].replace("\n", " ")
            logger.debug(
                "  [%d] %s | %s | %s | %s | %s…",
                i,
                c.get("document_name", "?"),
                c.get("source_type", "?"),
                score_str,
                c.get("retrieval_type", "dense"),
                preview,
            )

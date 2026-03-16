"""
Chat service for CVS Chatbot.
Orchestrates RAG pipeline: retrieve relevant context, build prompt, generate response.
"""

import json
import logging
from typing import Any, AsyncGenerator

from backend.config import Settings
from backend.services.openai_service import OpenAIService
from backend.services.vector_db import QdrantService

logger = logging.getLogger(__name__)

CVS_SYSTEM_PROMPT = (
    "You are a helpful assistant for Central Ventilation System (CVS), "
    "a professional ventilation company. Answer questions based on the provided "
    "document context. If the information is not available in the context, say so "
    "politely and suggest contacting CVS support for further assistance. "
    "Be professional, clear, and concise. When referencing specific information, "
    "mention the source document and page if available."
)


class ChatService:
    """Handles the end-to-end RAG chat pipeline."""

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
    # Non-streaming (returns full answer at once)
    # ------------------------------------------------------------------

    async def get_response(
        self,
        message: str,
        conversation_history: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        if conversation_history is None:
            conversation_history = []

        logger.info("Chat request: %.80s", message)

        query_vector = await self.openai.generate_embedding(message)
        chunks = self.qdrant.search(query_vector=query_vector, top_k=self.config.top_k_results)
        logger.info("Retrieved %d chunks from Qdrant.", len(chunks))

        context = self._build_context(chunks)
        messages = self._build_messages(message, context, conversation_history)
        answer = await self.openai.chat_completion(messages, self.config.max_tokens)
        sources = self._deduplicate_sources(chunks)

        return {"answer": answer, "sources": sources}

    # ------------------------------------------------------------------
    # Streaming — all blocking work done BEFORE generator starts
    # ------------------------------------------------------------------

    async def get_response_stream(
        self,
        message: str,
        conversation_history: list[dict[str, Any]] | None = None,
    ) -> AsyncGenerator[str, None]:
        """
        Prepares the full context, then returns an async generator that
        streams SSE-formatted tokens.

        Separating preparation from streaming ensures the synchronous
        Qdrant call never blocks inside the generator.
        """
        if conversation_history is None:
            conversation_history = []

        # --- All blocking / async prep work done here (outside the generator) ---
        query_vector = await self.openai.generate_embedding(message)
        chunks = self.qdrant.search(query_vector=query_vector, top_k=self.config.top_k_results)
        context = self._build_context(chunks)
        messages = self._build_messages(message, context, conversation_history)
        sources = self._deduplicate_sources(chunks)

        # --- Return a pure async generator that only does OpenAI streaming ---
        return self._stream_tokens(messages, sources)

    async def _stream_tokens(
        self,
        messages: list[dict[str, Any]],
        sources: list[dict[str, Any]],
    ) -> AsyncGenerator[str, None]:
        """Pure async generator: yields SSE events from OpenAI token stream."""
        yield f"data: {json.dumps({'type': 'sources', 'sources': sources})}\n\n"

        async for token in self.openai.chat_completion_stream(
            messages=messages,
            max_tokens=self.config.max_tokens,
        ):
            yield f"data: {json.dumps({'type': 'token', 'content': token})}\n\n"

        yield 'data: {"type":"done"}\n\n'

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_context(self, chunks: list[dict]) -> str:
        if not chunks:
            return "No relevant documentation found."
        parts = [
            f"[Source {i}: {c.get('document_name', 'unknown')}, Page {c.get('page_number', '?')}]\n{c.get('text_chunk', '')}"
            for i, c in enumerate(chunks, 1)
        ]
        return "\n\n---\n\n".join(parts)

    def _build_messages(
        self,
        question: str,
        context: str,
        conversation_history: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": f"{CVS_SYSTEM_PROMPT}\n\nDOCUMENT CONTEXT:\n{context}"}
        ]
        for entry in (conversation_history[-6:] if conversation_history else []):
            role = entry.get("role", "user")
            content = entry.get("content", "")
            if role in ("user", "assistant") and content:
                messages.append({"role": role, "content": content})
        messages.append({"role": "user", "content": question})
        return messages

    def _deduplicate_sources(self, chunks: list[dict]) -> list[dict]:
        seen: set[tuple] = set()
        sources: list[dict] = []
        for chunk in chunks:
            key = (chunk.get("document_name"), chunk.get("page_number"))
            if key not in seen:
                seen.add(key)
                sources.append({
                    "document_name": chunk.get("document_name", "unknown"),
                    "page_number": chunk.get("page_number", 0),
                    "score": chunk.get("score", 0.0),
                })
        return sources

"""
RAG retriever for CVS Chatbot.
Combines embedding generation with Qdrant vector search.
"""

import logging

from backend.config import Settings
from backend.services.openai_service import OpenAIService
from backend.services.vector_db import QdrantService

logger = logging.getLogger(__name__)


class RAGRetriever:
    """
    Retrieval-Augmented Generation retriever.
    Embeds a query and fetches the most relevant document chunks from Qdrant.
    """

    def __init__(
        self,
        qdrant_service: QdrantService,
        openai_service: OpenAIService,
        config: Settings,
    ):
        self.qdrant = qdrant_service
        self.openai = openai_service
        self.config = config

    async def retrieve(self, query: str) -> list[dict]:
        """
        Retrieve the top-k most relevant chunks for the given query.

        Args:
            query: Natural-language question or search string.

        Returns:
            List of chunk dicts (text_chunk, document_name, page_number, score).
        """
        logger.debug("Retrieving context for query: %.80s", query)
        query_vector = await self.openai.generate_embedding(query)
        chunks = self.qdrant.search(
            query_vector=query_vector,
            top_k=self.config.top_k_results,
        )
        logger.debug("Retrieved %d chunks.", len(chunks))
        return chunks

    def format_context(self, chunks: list[dict]) -> str:
        """
        Format a list of retrieved chunks into a single context string
        suitable for inclusion in an LLM prompt.

        Args:
            chunks: List of chunk dicts as returned by `retrieve`.

        Returns:
            Formatted context string.
        """
        if not chunks:
            return "No relevant context found in the document store."

        parts: list[str] = []
        for i, chunk in enumerate(chunks, start=1):
            doc = chunk.get("document_name", "unknown")
            page = chunk.get("page_number", "?")
            score = chunk.get("score", 0.0)
            text = chunk.get("text_chunk", "")
            parts.append(
                f"[Chunk {i} | Document: {doc} | Page: {page} | Relevance: {score:.2f}]\n{text}"
            )

        return "\n\n" + "\n\n---\n\n".join(parts) + "\n"

"""
OpenAI service for CVS Chatbot.
Handles embedding generation and chat completions.
"""

import asyncio
import logging
from typing import Any

from openai import AsyncOpenAI, OpenAI

from backend.config import Settings

logger = logging.getLogger(__name__)


class OpenAIService:
    """Wrapper around the OpenAI API for embeddings and chat completions."""

    def __init__(self, config: Settings):
        self.config = config
        self._async_client = AsyncOpenAI(api_key=config.openai_api_key)
        self._sync_client = OpenAI(api_key=config.openai_api_key)

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
        """
        Generate embeddings for a list of texts.
        Processes in batches of 100 to respect API limits.
        """
        all_embeddings: list[list[float]] = []
        batch_size = 100

        for i in range(0, len(texts), batch_size):
            batch = [t.replace("\n", " ").strip() for t in texts[i : i + batch_size]]
            response = await self._async_client.embeddings.create(
                input=batch,
                model=self.config.openai_embedding_model,
            )
            # Ensure order is preserved
            sorted_data = sorted(response.data, key=lambda x: x.index)
            all_embeddings.extend([item.embedding for item in sorted_data])
            logger.debug(
                "Generated embeddings for batch %d-%d", i, i + len(batch) - 1
            )

        return all_embeddings

    def generate_embedding_sync(self, text: str) -> list[float]:
        """Synchronous wrapper for generate_embedding."""
        return asyncio.get_event_loop().run_until_complete(
            self.generate_embedding(text)
        )

    def generate_embeddings_sync(self, texts: list[str]) -> list[list[float]]:
        """Synchronous wrapper for generate_embeddings."""
        return asyncio.get_event_loop().run_until_complete(
            self.generate_embeddings(texts)
        )

    # ------------------------------------------------------------------
    # Chat Completions
    # ------------------------------------------------------------------

    async def chat_completion(
        self,
        messages: list[dict[str, Any]],
        max_tokens: int | None = None,
    ) -> str:
        """
        Generate a chat completion response.

        Args:
            messages: List of message dicts with 'role' and 'content' keys.
            max_tokens: Maximum tokens in the response.

        Returns:
            The assistant's response as a string.
        """
        if max_tokens is None:
            max_tokens = self.config.max_tokens

        response = await self._async_client.chat.completions.create(
            model=self.config.openai_chat_model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=0.3,
        )
        return response.choices[0].message.content or ""

    async def chat_completion_stream(
        self,
        messages: list[dict[str, Any]],
        max_tokens: int | None = None,
    ):
        """
        Stream a chat completion response (async generator).
        Yields text chunks as they arrive.
        """
        if max_tokens is None:
            max_tokens = self.config.max_tokens

        stream = await self._async_client.chat.completions.create(
            model=self.config.openai_chat_model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=0.3,
            stream=True,
        )
        async for chunk in stream:
            content = chunk.choices[0].delta.content if chunk.choices else None
            if content:
                yield content

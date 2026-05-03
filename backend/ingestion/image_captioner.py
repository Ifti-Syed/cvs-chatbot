"""
GPT-4o Vision captioning for CF761 figure images.

Each image is sent to GPT-4o Vision and receives a detailed text description.
That description becomes the embedding text for the image's Qdrant point,
enabling semantic search over visual content without multimodal embeddings.

The image is also re-uploaded to the 'images/{document_name}/' GCS namespace
so the existing /api/images proxy can serve it to the frontend unchanged.
"""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


_SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

_MIME_MAP = {
    ".jpg":  "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png":  "image/png",
    ".webp": "image/webp",
}


class ImageCaptioner:
    """Generates searchable text descriptions of images using GPT-4o Vision."""

    def __init__(self, openai_service):
        self._openai = openai_service

    async def caption(self, image_bytes: bytes, filename: str) -> str:
        """Return a GPT-4o Vision description for the given image bytes."""
        return await self._openai.describe_image(image_bytes, filename)

    @staticmethod
    def is_supported(filename: str) -> bool:
        return Path(filename).suffix.lower() in _SUPPORTED_EXTENSIONS

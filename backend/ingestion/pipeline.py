"""
Ingestion pipeline.

Orchestrates: PDF source (local file or GCS) → text extraction →
chunking → embedding generation → Qdrant vector storage.

Two entry points:
  - process_pdf()          – ingest from a local file path
  - process_pdf_from_gcs() – download from GCS then ingest
"""

import logging
import os
import re
import tempfile
import uuid
from datetime import datetime, timezone
from typing import Optional

from backend.config import Settings
from backend.ingestion.pdf_processor import PDFProcessor
from backend.services.openai_service import OpenAIService
from backend.services.vector_db import QdrantService

logger = logging.getLogger(__name__)

# Batch size for Qdrant upserts — keeps each HTTP payload under ~10 MB
_UPSERT_BATCH = 200


class IngestionPipeline:
    """End-to-end pipeline: PDF → chunks → embeddings → Qdrant."""

    def __init__(
        self,
        qdrant_service: QdrantService,
        openai_service: OpenAIService,
        config: Settings,
        gcs_service=None,
    ):
        self.qdrant = qdrant_service
        self.openai = openai_service
        self.config = config
        self.gcs = gcs_service
        self.pdf_processor = PDFProcessor()

    # ------------------------------------------------------------------
    # Public entry points
    # ------------------------------------------------------------------

    async def process_pdf(
        self,
        file_path: str,
        document_name: str,
        source_folder: Optional[str] = None,
        upload_timestamp: Optional[str] = None,
    ) -> dict:
        logger.info(
            "Ingestion START | document='%s' | source='%s'", document_name, file_path
        )
        source_folder = source_folder or "local_upload"
        upload_timestamp = upload_timestamp or datetime.now(timezone.utc).isoformat()
        return await self._run_pipeline(
            file_path=file_path,
            document_name=document_name,
            source_folder=source_folder,
            upload_timestamp=upload_timestamp,
        )

    async def process_pdf_from_gcs(
        self,
        blob_name: str,
        document_name: Optional[str] = None,
    ) -> dict:
        if self.gcs is None:
            raise RuntimeError(
                "GCSService is not configured. "
                "Set GCS_BUCKET_NAME and GCS_FOLDER_PATH in your environment."
            )

        filename = blob_name.split("/")[-1]
        if document_name is None:
            document_name = filename[:-4] if filename.lower().endswith(".pdf") else filename

        parts = blob_name.rsplit("/", 1)
        source_folder = (
            f"gs://{self.gcs.bucket_name}/{parts[0]}"
            if len(parts) > 1
            else f"gs://{self.gcs.bucket_name}"
        )

        upload_timestamp = self.gcs.get_upload_timestamp(blob_name) or datetime.now(
            timezone.utc
        ).isoformat()

        logger.info(
            "Ingestion START (GCS) | document='%s' | blob='%s'",
            document_name,
            blob_name,
        )

        pdf_bytes = self.gcs.download_pdf(blob_name)
        tmp_path = ""
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(pdf_bytes)
                tmp_path = tmp.name
            return await self._run_pipeline(
                file_path=tmp_path,
                document_name=document_name,
                source_folder=source_folder,
                upload_timestamp=upload_timestamp,
            )
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)

    # ------------------------------------------------------------------
    # Core pipeline
    # ------------------------------------------------------------------

    async def _run_pipeline(
        self,
        file_path: str,
        document_name: str,
        source_folder: str,
        upload_timestamp: str,
    ) -> dict:
        # Step 1 — extract pages
        pages = self.pdf_processor.extract_text(file_path)
        if not pages:
            raise ValueError(f"No extractable text found in '{file_path}'.")

        # Step 2 — chunk all pages
        all_chunks: list[dict] = []
        global_chunk_index = 0
        for page_data in pages:
            page_chunks = self._chunk_text(
                text=page_data["text"],
                page=page_data["page"],
            )
            for chunk in page_chunks:
                chunk["chunk_index"] = global_chunk_index
                global_chunk_index += 1
            all_chunks.extend(page_chunks)

        if not all_chunks:
            raise ValueError("No text chunks produced from document.")

        logger.info(
            "Document '%s': %d page(s) → %d chunk(s)",
            document_name,
            len(pages),
            len(all_chunks),
        )

        # Step 3 — batch-generate embeddings
        texts = [c["text"] for c in all_chunks]
        embeddings = await self.openai.generate_embeddings(texts)

        # Step 4 — build Qdrant point dicts
        vectors: list[dict] = []
        for chunk, embedding in zip(all_chunks, embeddings):
            vectors.append(
                {
                    "id": str(uuid.uuid4()),
                    "vector": embedding,
                    "payload": {
                        "document_name": document_name,
                        "page_number": chunk["page"],
                        "text_chunk": chunk["text"],
                        "chunk_index": chunk["chunk_index"],
                        "source_folder": source_folder,
                        "upload_timestamp": upload_timestamp,
                    },
                }
            )

        # Step 5 — upsert into Qdrant in batches
        for i in range(0, len(vectors), _UPSERT_BATCH):
            self.qdrant.upsert_vectors(vectors[i: i + _UPSERT_BATCH])

        logger.info(
            "Ingestion DONE | document='%s' | chunks=%d | pages=%d",
            document_name,
            len(vectors),
            len(pages),
        )
        return {
            "document_name": document_name,
            "total_chunks": len(vectors),
            "pages": len(pages),
            "source_folder": source_folder,
            "upload_timestamp": upload_timestamp,
        }

    # ------------------------------------------------------------------
    # Text chunking
    # ------------------------------------------------------------------

    def _chunk_text(self, text: str, page: int) -> list[dict]:
        """
        Split page text into overlapping chunks with sentence-boundary awareness.

        Improvements over naive character splitting:
          - Respects paragraph > sentence > word boundary hierarchy.
          - Drops chunks shorter than min_chunk_size to avoid orphan fragments.
          - Normalises whitespace before splitting so boundary detection is reliable.
        """
        chunk_size = self.config.chunk_size
        overlap = self.config.chunk_overlap
        min_size = self.config.min_chunk_size

        # Normalise: collapse excessive blank lines but preserve paragraph breaks
        text = re.sub(r"\n{3,}", "\n\n", text).strip()

        if len(text) <= chunk_size:
            return [{"text": text, "page": page}] if len(text) >= min_size else []

        chunks: list[dict] = []
        start = 0

        while start < len(text):
            end = start + chunk_size

            if end >= len(text):
                chunk = text[start:].strip()
                if len(chunk) >= min_size:
                    chunks.append({"text": chunk, "page": page})
                break

            split_pos = self._find_split_position(text, start, end)
            chunk = text[start:split_pos].strip()

            if len(chunk) >= min_size:
                chunks.append({"text": chunk, "page": page})
            else:
                logger.debug(
                    "Skipped short chunk (%d chars) on page %d.", len(chunk), page
                )

            # Advance with overlap so boundary context is preserved
            next_start = max(split_pos - overlap, start + 1)
            start = next_start

        return chunks

    @staticmethod
    def _find_split_position(text: str, start: int, end: int) -> int:
        """
        Find the best split position at or before `end`.

        Priority:
            1. Paragraph break (double newline)
            2. Single newline
            3. Sentence-ending punctuation followed by a space
            4. Word boundary (space)
            5. Hard cut at `end`
        """
        # Search in a window behind end so we don't backtrack too far
        search_start = max(start, end - 300)
        segment = text[search_start:end]

        # 1. Paragraph break
        pos = segment.rfind("\n\n")
        if pos != -1:
            return search_start + pos + 2

        # 2. Single newline
        pos = segment.rfind("\n")
        if pos != -1:
            return search_start + pos + 1

        # 3. Sentence-ending punctuation
        for punct in (". ", "! ", "? ", ".\n", "!\n", "?\n"):
            pos = segment.rfind(punct)
            if pos != -1:
                return search_start + pos + len(punct)

        # 4. Word boundary
        pos = segment.rfind(" ")
        if pos != -1:
            return search_start + pos + 1

        # 5. Hard cut
        return end

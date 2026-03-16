"""
Ingestion pipeline for CVS Chatbot.

Orchestrates: PDF source (local file or GCS) → text extraction →
chunking → embedding generation → Qdrant vector storage.

Two entry points are provided:
  - process_pdf()          – ingest from a local file path (used by the
                             HTTP upload endpoint and testing)
  - process_pdf_from_gcs() – download from GCS then ingest (used by the
                             GCS-trigger endpoints)
"""

import logging
import os
import tempfile
import uuid
from datetime import datetime, timezone
from typing import Optional

from backend.config import Settings
from backend.ingestion.pdf_processor import PDFProcessor
from backend.services.openai_service import OpenAIService
from backend.services.vector_db import QdrantService

logger = logging.getLogger(__name__)

# Batch size for Qdrant upserts – keeps each HTTP payload under ~10 MB
_UPSERT_BATCH = 200


class IngestionPipeline:
    """End-to-end pipeline: PDF → chunks → embeddings → Qdrant."""

    def __init__(
        self,
        qdrant_service: QdrantService,
        openai_service: OpenAIService,
        config: Settings,
        gcs_service=None,  # Optional[GCSService] – injected when GCS is configured
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
        """
        Full ingestion pipeline for a PDF at a local file path.

        Args:
            file_path:        Absolute path to the PDF file on disk.
            document_name:    Logical name stored as metadata (no extension).
            source_folder:    Optional GCS folder / bucket path for metadata.
            upload_timestamp: Optional ISO-8601 timestamp for metadata.

        Returns:
            dict with keys: document_name, total_chunks, pages
        """
        logger.info(
            "Ingestion START | document='%s' | source='%s'",
            document_name,
            file_path,
        )

        # Resolve metadata defaults
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
        """
        Download a PDF from GCS and run the full ingestion pipeline.

        The original PDF is NOT deleted from GCS (kept for backup/re-embed).

        Args:
            blob_name:     Full GCS object path (e.g. "sales_pdfs/catalog.pdf").
            document_name: Override the logical name; defaults to the bare
                           filename without the .pdf extension.

        Returns:
            dict with keys: document_name, total_chunks, pages
        """
        if self.gcs is None:
            raise RuntimeError(
                "GCSService is not configured. "
                "Set GCS_BUCKET_NAME and GCS_FOLDER_PATH in your environment."
            )

        # Derive document name from blob path when not explicitly provided
        filename = blob_name.split("/")[-1]
        if document_name is None:
            document_name = filename[:-4] if filename.lower().endswith(".pdf") else filename

        # Derive source folder from blob path (everything before the filename)
        parts = blob_name.rsplit("/", 1)
        source_folder = (
            f"gs://{self.gcs.bucket_name}/{parts[0]}" if len(parts) > 1
            else f"gs://{self.gcs.bucket_name}"
        )

        # Get the upload timestamp from GCS metadata
        upload_timestamp = self.gcs.get_upload_timestamp(blob_name) or datetime.now(
            timezone.utc
        ).isoformat()

        logger.info(
            "Ingestion START (GCS) | document='%s' | blob='%s'",
            document_name,
            blob_name,
        )

        # Download to a temporary file so PDFProcessor can use its normal path-based API
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
    # Core pipeline (shared by both entry points)
    # ------------------------------------------------------------------

    async def _run_pipeline(
        self,
        file_path: str,
        document_name: str,
        source_folder: str,
        upload_timestamp: str,
    ) -> dict:
        """
        Extract → chunk → embed → store.

        All chunks are tagged with document_name, page_number,
        source_folder, and upload_timestamp in their Qdrant payload.
        """
        # Step 1 – extract pages
        pages = self.pdf_processor.extract_text(file_path)
        if not pages:
            raise ValueError(f"No extractable text found in '{file_path}'.")

        # Step 2 – chunk all pages
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

        # Step 3 – batch-generate embeddings (100 texts per OpenAI request)
        texts = [c["text"] for c in all_chunks]
        embeddings = await self.openai.generate_embeddings(texts)

        # Step 4 – build Qdrant point dicts
        vectors: list[dict] = []
        for chunk, embedding in zip(all_chunks, embeddings):
            vectors.append(
                {
                    "id": str(uuid.uuid4()),
                    "vector": embedding,
                    "payload": {
                        # Core retrieval metadata
                        "document_name": document_name,
                        "page_number": chunk["page"],
                        "text_chunk": chunk["text"],
                        "chunk_index": chunk["chunk_index"],
                        # Provenance metadata
                        "source_folder": source_folder,
                        "upload_timestamp": upload_timestamp,
                    },
                }
            )

        # Step 5 – upsert into Qdrant in batches
        for i in range(0, len(vectors), _UPSERT_BATCH):
            self.qdrant.upsert_vectors(vectors[i : i + _UPSERT_BATCH])

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
    # Text chunking helpers
    # ------------------------------------------------------------------

    def _chunk_text(self, text: str, page: int) -> list[dict]:
        """
        Split a block of text into overlapping chunks with sentence-boundary
        awareness.

        Args:
            text: The page text to split.
            page: 1-based page number (stored as metadata).

        Returns:
            List of dicts with 'text' (str) and 'page' (int).
        """
        chunk_size = self.config.chunk_size
        overlap = self.config.chunk_overlap

        if len(text) <= chunk_size:
            return [{"text": text, "page": page}]

        chunks: list[dict] = []
        start = 0

        while start < len(text):
            end = start + chunk_size

            if end >= len(text):
                chunk = text[start:].strip()
                if chunk:
                    chunks.append({"text": chunk, "page": page})
                break

            split_pos = self._find_split_position(text, start, end)
            chunk = text[start:split_pos].strip()
            if chunk:
                chunks.append({"text": chunk, "page": page})

            # Advance with overlap so context at boundaries is preserved
            start = max(split_pos - overlap, start + 1)

        return chunks

    @staticmethod
    def _find_split_position(text: str, start: int, end: int) -> int:
        """
        Find the best position to split text before index `end`.

        Priority order:
            1. Paragraph break (double newline)
            2. Single newline
            3. Sentence end (". ", "! ", "? ")
            4. Word boundary (space)
            5. Hard cut at `end`
        """
        search_start = max(start, end - 256)
        segment = text[search_start:end]

        para_pos = segment.rfind("\n\n")
        if para_pos != -1:
            return search_start + para_pos + 2

        newline_pos = segment.rfind("\n")
        if newline_pos != -1:
            return search_start + newline_pos + 1

        for punct in (". ", "! ", "? "):
            sent_pos = segment.rfind(punct)
            if sent_pos != -1:
                return search_start + sent_pos + len(punct)

        space_pos = segment.rfind(" ")
        if space_pos != -1:
            return search_start + space_pos + 1

        return end

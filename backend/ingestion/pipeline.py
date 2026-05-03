"""
Ingestion pipeline — orchestrates document processing end-to-end.

Supported file types:
  PDF   → PDFProcessor  (text + tables via pdfplumber)
           + optional page-image extraction (requires pymupdf + GCS)
  Excel → ExcelProcessor (sheet → pipe-delimited text via pandas)

Flow:
  source (local file or GCS blob)
    → processor (PDF or Excel)
    → text chunking
    → embedding generation (OpenAI)
    → Qdrant upsert

Image extraction (PDF only, non-blocking):
  When pymupdf is installed and GCS is configured each PDF page is rendered
  to PNG and uploaded to GCS under  images/<document_name>/page_<N>.png.
  The GCS-relative blob name is stored in the Qdrant payload so the API
  can serve it via the /api/images/… proxy endpoint.
  Pages with no extractable text but with a page image get a synthetic text
  chunk so they are still discoverable via semantic search.
"""

import logging
import os
import re
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from backend.config import Settings
from backend.ingestion.excel_processor import EXCEL_EXTENSIONS, ExcelProcessor
from backend.ingestion.image_captioner import ImageCaptioner
from backend.ingestion.json_processor import JSONTableProcessor
from backend.ingestion.pdf_processor import PDFProcessor
from backend.services.openai_service import OpenAIService
from backend.services.vector_db import QdrantService

import re as _re

logger = logging.getLogger(__name__)

# Batch size for Qdrant upserts — keeps each HTTP payload under ~10 MB
_UPSERT_BATCH = 200

_FIGURE_NUM_RE = _re.compile(r'figure\s*(\d+)', _re.IGNORECASE)
_APPENDIX_RE   = _re.compile(r'append[ie]x\s*([a-z])', _re.IGNORECASE)


def _parse_figure_info(filename: str) -> tuple[str, int | None]:
    """
    Return (figure_name, figure_number) from an image filename.

    "CF761-Figure 3.jpg"  →  ("Figure 3",  3)
    "Appendex B.jpg"      →  ("Appendix B", None)
    "other.jpg"           →  ("other",      None)
    """
    stem = Path(filename).stem
    m = _FIGURE_NUM_RE.search(stem)
    if m:
        n = int(m.group(1))
        return f"Figure {n}", n
    m = _APPENDIX_RE.search(stem)
    if m:
        return f"Appendix {m.group(1).upper()}", None
    return stem, None

# GCS sub-folder for extracted page images
_IMAGES_FOLDER = "images"


class IngestionPipeline:
    """End-to-end pipeline: document → chunks → embeddings → Qdrant."""

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
        self.excel_processor = ExcelProcessor()
        self.json_processor = JSONTableProcessor()
        self.image_captioner = ImageCaptioner(openai_service)

    # ------------------------------------------------------------------
    # Public entry points
    # ------------------------------------------------------------------

    async def process_pdf(
        self,
        file_path: str,
        document_name: str,
        source_folder: Optional[str] = None,
        upload_timestamp: Optional[str] = None,
        source_type_override: Optional[str] = None,
    ) -> dict:
        logger.info(
            "Ingestion START | document='%s' | source='%s'", document_name, file_path
        )
        return await self._run_pipeline(
            file_path=file_path,
            document_name=document_name,
            source_folder=source_folder or "local_upload",
            upload_timestamp=upload_timestamp or datetime.now(timezone.utc).isoformat(),
            file_type="pdf",
            source_type_override=source_type_override,
        )

    async def process_excel(
        self,
        file_path: str,
        document_name: str,
        source_folder: Optional[str] = None,
        upload_timestamp: Optional[str] = None,
    ) -> dict:
        logger.info(
            "Ingestion START (Excel) | document='%s' | source='%s'",
            document_name, file_path,
        )
        return await self._run_pipeline(
            file_path=file_path,
            document_name=document_name,
            source_folder=source_folder or "local_upload",
            upload_timestamp=upload_timestamp or datetime.now(timezone.utc).isoformat(),
            file_type="excel",
        )

    async def process_pdf_from_gcs(
        self,
        blob_name: str,
        document_name: Optional[str] = None,
        source_type_override: Optional[str] = None,
    ) -> dict:
        return await self._process_from_gcs(
            blob_name, document_name, file_type="pdf",
            source_type_override=source_type_override,
        )

    async def process_excel_from_gcs(
        self,
        blob_name: str,
        document_name: Optional[str] = None,
    ) -> dict:
        return await self._process_from_gcs(blob_name, document_name, file_type="excel")

    async def _process_from_gcs(
        self,
        blob_name: str,
        document_name: Optional[str],
        file_type: str,
        source_type_override: Optional[str] = None,
    ) -> dict:
        if self.gcs is None:
            raise RuntimeError(
                "GCSService is not configured. "
                "Set GCS_BUCKET_NAME and GCS_FOLDER_PATH in your environment."
            )

        filename = blob_name.split("/")[-1]
        if document_name is None:
            stem = Path(filename).stem
            document_name = stem

        parts = blob_name.rsplit("/", 1)
        source_folder = (
            f"gs://{self.gcs.bucket_name}/{parts[0]}"
            if len(parts) > 1
            else f"gs://{self.gcs.bucket_name}"
        )

        upload_timestamp = (
            self.gcs.get_upload_timestamp(blob_name)
            or datetime.now(timezone.utc).isoformat()
        )

        logger.info(
            "Ingestion START (GCS %s) | document='%s' | blob='%s'",
            file_type.upper(), document_name, blob_name,
        )

        file_bytes = self.gcs.download_file(blob_name)
        suffix = Path(filename).suffix or f".{file_type}"
        tmp_path = ""
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(file_bytes)
                tmp_path = tmp.name
            return await self._run_pipeline(
                file_path=tmp_path,
                document_name=document_name,
                source_folder=source_folder,
                upload_timestamp=upload_timestamp,
                file_type=file_type,
                source_type_override=source_type_override,
            )
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)

    # ------------------------------------------------------------------
    # JSON table ingestion
    # ------------------------------------------------------------------

    async def process_json_local(
        self,
        file_path: str,
        document_name: Optional[str] = None,
    ) -> dict:
        """Ingest a local JSON table file into Qdrant."""
        filename = Path(file_path).name
        doc_name = document_name or Path(file_path).stem
        source_folder = f"local/{filename}"
        upload_timestamp = datetime.now(timezone.utc).isoformat()

        logger.info("JSON ingestion START | document='%s' | path='%s'", doc_name, file_path)
        records = self.json_processor.load_file(file_path)
        pages = self.json_processor.records_to_pages(records, filename)
        return await self._run_json_pipeline(pages, doc_name, source_folder, upload_timestamp)

    async def process_json_from_gcs(
        self,
        blob_name: str,
        document_name: Optional[str] = None,
    ) -> dict:
        """Download a JSON file from GCS and ingest it into Qdrant."""
        if self.gcs is None:
            raise RuntimeError("GCSService is not configured.")

        filename = blob_name.split("/")[-1]
        doc_name = document_name or Path(filename).stem
        source_folder = f"gs://{self.gcs.bucket_name}/{blob_name.rsplit('/', 1)[0]}"
        upload_timestamp = (
            self.gcs.get_upload_timestamp(blob_name)
            or datetime.now(timezone.utc).isoformat()
        )

        logger.info("JSON ingestion START (GCS) | document='%s' | blob='%s'", doc_name, blob_name)
        file_bytes = self.gcs.download_file(blob_name)

        tmp_path = ""
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as tmp:
                tmp.write(file_bytes)
                tmp_path = tmp.name
            records = self.json_processor.load_file(tmp_path)
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)

        pages = self.json_processor.records_to_pages(records, filename)
        return await self._run_json_pipeline(pages, doc_name, source_folder, upload_timestamp)

    async def _run_json_pipeline(
        self,
        pages: list[dict],
        document_name: str,
        source_folder: str,
        upload_timestamp: str,
    ) -> dict:
        """Embed and upsert JSON table records into Qdrant.

        Each page dict is a single pre-chunked record — no further splitting.
        All JSON-specific fields (table_number, duct_size, etc.) are stored in
        the Qdrant payload for precise filtering.
        """
        if not pages:
            raise ValueError(f"No records to ingest for '{document_name}'.")

        texts = [p["text"] for p in pages]
        embeddings = await self.openai.generate_embeddings(texts)

        vectors: list[dict] = []
        for idx, (page, embedding) in enumerate(zip(pages, embeddings)):
            payload: dict = {
                "document_name":    document_name,
                "page_number":      page["page"],
                "text_chunk":       page["text"],
                "chunk_index":      idx,
                "source_folder":    source_folder,
                "upload_timestamp": upload_timestamp,
                "file_type":        "json",
                "source_type":      "json_table",
                "image_blob_names": [],
                # JSON-specific fields
                "table_number":      page.get("table_number", ""),
                "table_description": page.get("table_description", ""),
                "title":             page.get("title", ""),
                "system":            page.get("system", ""),
                "insulation":        page.get("insulation", ""),
                "duct_size":         page.get("duct_size", ""),
                "source_file":       page.get("source_file", ""),
                "json_metadata":     page.get("json_metadata", {}),
            }
            vectors.append({
                "id":      str(uuid.uuid4()),
                "vector":  embedding,
                "payload": payload,
            })

        for i in range(0, len(vectors), _UPSERT_BATCH):
            self.qdrant.upsert_vectors(vectors[i: i + _UPSERT_BATCH])

        logger.info(
            "JSON ingestion DONE | document='%s' | records=%d",
            document_name, len(vectors),
        )
        return {
            "document_name": document_name,
            "total_chunks":  len(vectors),
            "pages":         len(pages),
            "source_folder": source_folder,
            "upload_timestamp": upload_timestamp,
        }

    # ------------------------------------------------------------------
    # Image captioning ingestion
    # ------------------------------------------------------------------

    async def process_images_local(
        self,
        folder_path: str,
        document_name: str = "CF761",
    ) -> dict:
        """Caption all images in a local folder and ingest into Qdrant."""
        from backend.ingestion.image_captioner import ImageCaptioner as _IC
        folder = Path(folder_path)
        if not folder.exists():
            raise FileNotFoundError(f"Image folder not found: {folder_path}")

        _SUPPORTED = {".jpg", ".jpeg", ".png", ".webp"}
        image_files = [f for f in sorted(folder.iterdir()) if f.suffix.lower() in _SUPPORTED]
        if not image_files:
            raise ValueError(f"No supported images found in '{folder_path}'.")

        upload_timestamp = datetime.now(timezone.utc).isoformat()
        source_folder = f"local/{folder.name}"

        logger.info(
            "Image ingestion START (local) | document='%s' | images=%d",
            document_name, len(image_files),
        )

        image_items: list[dict] = []
        for img_path in image_files:
            image_bytes = img_path.read_bytes()
            image_items.append({
                "filename":    img_path.name,
                "bytes":       image_bytes,
                "source_blob": str(img_path),
            })

        return await self._run_image_pipeline(
            image_items=image_items,
            document_name=document_name,
            source_folder=source_folder,
            upload_timestamp=upload_timestamp,
        )

    async def process_images_from_gcs(
        self,
        subfolder: str,
        document_name: str = "CF761",
    ) -> dict:
        """Download all images from a GCS subfolder, caption them, and ingest."""
        if self.gcs is None:
            raise RuntimeError("GCSService is not configured.")

        image_list = self.gcs.list_image_files(subfolder)
        if not image_list:
            raise ValueError(
                f"No images found under gs://{self.gcs.bucket_name}/"
                f"{self.gcs.folder_path}/{subfolder}"
            )

        upload_timestamp = datetime.now(timezone.utc).isoformat()
        source_folder = (
            f"gs://{self.gcs.bucket_name}/{self.gcs.folder_path}/{subfolder.strip('/')}"
        )

        logger.info(
            "Image ingestion START (GCS) | document='%s' | images=%d | subfolder='%s'",
            document_name, len(image_list), subfolder,
        )

        image_items: list[dict] = []
        for info in image_list:
            try:
                image_bytes = self.gcs.download_file(info["blob_name"])
                image_items.append({
                    "filename":    info["filename"],
                    "bytes":       image_bytes,
                    "source_blob": info["blob_name"],
                })
            except Exception as exc:
                logger.warning(
                    "Could not download image '%s': %s — skipping.", info["blob_name"], exc
                )

        if not image_items:
            raise ValueError("No images could be downloaded from GCS.")

        return await self._run_image_pipeline(
            image_items=image_items,
            document_name=document_name,
            source_folder=source_folder,
            upload_timestamp=upload_timestamp,
        )

    async def _run_image_pipeline(
        self,
        image_items: list[dict],
        document_name: str,
        source_folder: str,
        upload_timestamp: str,
    ) -> dict:
        """Caption images, embed the captions, and upsert into Qdrant.

        For each image:
          1. Call GPT-4o Vision to generate a detailed text caption.
          2. Upload the image to images/{document_name}/{filename} in GCS
             (so the existing /api/images proxy can serve it to the frontend).
          3. Embed the caption and upsert a Qdrant point with source_type='image_caption'.
        """
        vectors: list[dict] = []

        for idx, item in enumerate(image_items):
            filename    = item["filename"]
            image_bytes = item["bytes"]
            source_blob = item.get("source_blob", "")

            # Step 1 — Generate caption
            try:
                caption = await self.image_captioner.caption(image_bytes, filename)
            except Exception as exc:
                logger.warning("Caption failed for '%s': %s — skipping.", filename, exc)
                continue

            # Step 2 — Copy image to the images/<doc>/<file> GCS namespace so the
            # existing frontend proxy (/api/images/{doc}/{file}) can serve it.
            proxy_blob = f"images/{document_name}/{filename}"
            if self.gcs is not None:
                try:
                    self.gcs.upload_image(image_bytes, proxy_blob)
                except Exception as exc:
                    logger.warning(
                        "Could not upload image '%s' to GCS proxy path: %s", filename, exc
                    )
                    proxy_blob = ""

            # Step 3 — Embed caption
            try:
                embedding = await self.openai.generate_embedding(caption)
            except Exception as exc:
                logger.warning("Embedding failed for '%s': %s — skipping.", filename, exc)
                continue

            figure_name, figure_number = _parse_figure_info(filename)
            payload: dict = {
                "document_name":    document_name,
                "page_number":      idx + 1,
                "text_chunk":       caption,
                "chunk_index":      idx,
                "source_folder":    source_folder,
                "upload_timestamp": upload_timestamp,
                "file_type":        "image",
                "source_type":      "image_caption",
                "image_blob_names": [proxy_blob] if proxy_blob else [],
                "image_filename":   filename,
                "image_gcs_source": source_blob,
                "figure_name":      figure_name,
                "figure_number":    figure_number,
            }
            vectors.append({
                "id":      str(uuid.uuid4()),
                "vector":  embedding,
                "payload": payload,
            })
            logger.info(
                "Image captioned and embedded: '%s' (%d/%d)",
                filename, idx + 1, len(image_items),
            )

        if not vectors:
            raise ValueError(f"No images could be processed for '{document_name}'.")

        for i in range(0, len(vectors), _UPSERT_BATCH):
            self.qdrant.upsert_vectors(vectors[i: i + _UPSERT_BATCH])

        logger.info(
            "Image ingestion DONE | document='%s' | images=%d",
            document_name, len(vectors),
        )
        return {
            "document_name":  document_name,
            "total_chunks":   len(vectors),
            "pages":          len(vectors),
            "source_folder":  source_folder,
            "upload_timestamp": upload_timestamp,
        }

    # ------------------------------------------------------------------
    # Core pipeline
    # ------------------------------------------------------------------

    async def _run_pipeline(
        self,
        file_path: str,
        document_name: str,
        source_folder: str,
        upload_timestamp: str,
        file_type: str,  # "pdf" | "excel"
        source_type_override: Optional[str] = None,
    ) -> dict:

        # Step 1 — extract pages (text + optional structured metadata)
        if file_type == "excel":
            pages = self.excel_processor.extract_text(file_path)
        else:
            pages = self.pdf_processor.extract_text(file_path)

        # PDF page images are not extracted — diagrams are served exclusively
        # from the dedicated CF761-Images folder (image_caption source type).
        page_images: dict[int, list[str]] = {}

        # Sort pages by page/sheet number
        pages.sort(key=lambda p: p["page"])

        if not pages:
            raise ValueError(f"No extractable content found in '{file_path}'.")

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
                chunk["image_blob_names"] = page_images.get(page_data["page"], [])
                # Extra metadata for Excel sheets
                if "sheet_name" in page_data:
                    chunk["sheet_name"] = page_data["sheet_name"]
                global_chunk_index += 1
            all_chunks.extend(page_chunks)

        if not all_chunks:
            raise ValueError("No text chunks produced from document.")

        logger.info(
            "Document '%s': %d page(s)/sheet(s) → %d chunk(s)",
            document_name, len(pages), len(all_chunks),
        )

        # Step 3 — batch-generate embeddings
        texts = [c["text"] for c in all_chunks]
        embeddings = await self.openai.generate_embeddings(texts)

        # Step 4 — build Qdrant point dicts
        vectors: list[dict] = []
        for chunk, embedding in zip(all_chunks, embeddings):
            payload: dict = {
                "document_name": document_name,
                "page_number": chunk["page"],
                "text_chunk": chunk["text"],
                "chunk_index": chunk["chunk_index"],
                "source_folder": source_folder,
                "upload_timestamp": upload_timestamp,
                "file_type": file_type,
                "source_type": source_type_override or ("pdf_text" if file_type == "pdf" else "excel_text"),
                "image_blob_names": chunk.get("image_blob_names", []),
            }
            if "sheet_name" in chunk:
                payload["sheet_name"] = chunk["sheet_name"]

            vectors.append({
                "id": str(uuid.uuid4()),
                "vector": embedding,
                "payload": payload,
            })

        # Step 5 — upsert into Qdrant in batches
        for i in range(0, len(vectors), _UPSERT_BATCH):
            self.qdrant.upsert_vectors(vectors[i: i + _UPSERT_BATCH])

        logger.info(
            "Ingestion DONE | document='%s' | chunks=%d | pages=%d",
            document_name, len(vectors), len(pages),
        )
        return {
            "document_name": document_name,
            "total_chunks": len(vectors),
            "pages": len(pages),
            "source_folder": source_folder,
            "upload_timestamp": upload_timestamp,
        }

    # ------------------------------------------------------------------
    # Image extraction (PDF pages → PNG → GCS)
    # ------------------------------------------------------------------

    def _extract_page_images(
        self, file_path: str, document_name: str
    ) -> dict[int, list[str]]:
        """
        Render each PDF page as a PNG and upload it to GCS.

        Requires:  pymupdf (fitz) installed  AND  GCS configured.
        Returns:   { page_number (1-based): [gcs_blob_name, ...] }
        Returns {} silently if either requirement is unmet.
        """
        if self.gcs is None:
            return {}

        try:
            import fitz  # pymupdf
        except ImportError:
            logger.debug(
                "pymupdf not installed — page image extraction disabled. "
                "Install with: pip install pymupdf"
            )
            return {}

        page_images: dict[int, list[str]] = {}

        try:
            doc = fitz.open(file_path)
            for page_index in range(len(doc)):
                page_num = page_index + 1  # 1-based
                fitz_page = doc[page_index]

                # Render at 2× zoom for legibility
                mat = fitz.Matrix(2.0, 2.0)
                pix = fitz_page.get_pixmap(matrix=mat)
                img_bytes = pix.tobytes("png")

                # Only upload non-trivial pages (> 2 KB — avoids truly blank pages)
                if len(img_bytes) < 2 * 1024:
                    continue

                filename = f"page_{page_num}.png"
                blob_name = f"{_IMAGES_FOLDER}/{document_name}/{filename}"

                try:
                    self.gcs.upload_image(img_bytes, blob_name)
                    page_images.setdefault(page_num, []).append(blob_name)
                    logger.debug(
                        "Uploaded page image: gs://%s/%s",
                        self.gcs.bucket_name, blob_name,
                    )
                except Exception as exc:
                    logger.warning(
                        "Could not upload image for page %d of '%s': %s",
                        page_num, document_name, exc,
                    )

            doc.close()

        except Exception as exc:
            logger.warning(
                "Image extraction failed for '%s': %s", document_name, exc
            )

        if page_images:
            total_imgs = sum(len(v) for v in page_images.values())
            logger.info(
                "Extracted and uploaded %d page image(s) for '%s'.",
                total_imgs, document_name,
            )

        return page_images

    # ------------------------------------------------------------------
    # Text chunking
    # ------------------------------------------------------------------

    def _chunk_text(self, text: str, page: int) -> list[dict]:
        """
        Split page/sheet text into overlapping chunks with boundary awareness.

        Respects paragraph > sentence > word boundary hierarchy.
        Drops chunks shorter than min_chunk_size to avoid orphan fragments.

        Table header injection: if a split lands inside a pipe-delimited
        Markdown table, the header row and separator are automatically
        prepended to every continuation chunk so the LLM always knows what
        each column means — even when a large table is split across chunks.
        """
        chunk_size = self.config.chunk_size
        overlap = self.config.chunk_overlap
        min_size = self.config.min_chunk_size

        text = re.sub(r"\n{3,}", "\n\n", text).strip()

        if len(text) <= chunk_size:
            return [{"text": text, "page": page}] if len(text) >= min_size else []

        chunks: list[dict] = []
        start = 0

        while start < len(text):
            end = start + chunk_size

            if end >= len(text):
                chunk = text[start:].strip()
                chunk = self._inject_table_header(text, start, chunk)
                if len(chunk) >= min_size:
                    chunks.append({"text": chunk, "page": page})
                break

            split_pos = self._find_split_position(text, start, end)
            chunk = text[start:split_pos].strip()
            chunk = self._inject_table_header(text, start, chunk)

            if len(chunk) >= min_size:
                chunks.append({"text": chunk, "page": page})
            else:
                logger.debug(
                    "Skipped short chunk (%d chars) on page %d.", len(chunk), page
                )

            next_start = max(split_pos - overlap, start + 1)
            start = next_start

        return chunks

    @staticmethod
    def _inject_table_header(full_text: str, chunk_start: int, chunk: str) -> str:
        """
        Prepend a pipe-delimited table's header + separator to `chunk` when
        the chunk starts mid-table (i.e. a split landed inside the data rows).

        Detection logic
        ───────────────
        A chunk is "mid-table" when:
          • its first non-empty line begins with "|"  (a table cell), AND
          • that first line is not itself a header (the second line is not a
            separator like |---|---| ), AND
          • the first line is not the separator row itself.

        Recovery: scan backward in `full_text` before `chunk_start` to find
        the most recent separator row and the header row above it.
        """
        if not chunk:
            return chunk

        lines = chunk.split("\n")
        first_line = lines[0].strip()

        # Not inside a table at all
        if not first_line.startswith("|"):
            return chunk

        def _is_separator(line: str) -> bool:
            s = line.strip()
            return s.startswith("|") and "-" in s and all(c in "-| " for c in s)

        # Already starts at a separator row — no injection needed
        if _is_separator(first_line):
            return chunk

        # Chunk starts at a header row (second line is the separator)
        if len(lines) > 1 and _is_separator(lines[1]):
            return chunk

        # ── Mid-table data row: find the header above chunk_start ──────────
        preceding = full_text[:chunk_start]
        prec_lines = preceding.split("\n")

        sep_idx = None
        for i in range(len(prec_lines) - 1, -1, -1):
            if _is_separator(prec_lines[i]):
                sep_idx = i
                break

        if sep_idx is None or sep_idx == 0:
            return chunk  # Can't locate a separator — leave chunk as-is

        header_line = prec_lines[sep_idx - 1]
        if not header_line.strip().startswith("|"):
            return chunk  # Malformed table — leave as-is

        sep_line = prec_lines[sep_idx]
        injected = f"{header_line}\n{sep_line}\n{chunk}"
        logger.debug(
            "Injected table header into mid-table chunk at offset %d.", chunk_start
        )
        return injected

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
        search_start = max(start, end - 300)
        segment = text[search_start:end]

        pos = segment.rfind("\n\n")
        if pos != -1:
            return search_start + pos + 2

        pos = segment.rfind("\n")
        if pos != -1:
            return search_start + pos + 1

        for punct in (". ", "! ", "? ", ".\n", "!\n", "?\n"):
            pos = segment.rfind(punct)
            if pos != -1:
                return search_start + pos + len(punct)

        pos = segment.rfind(" ")
        if pos != -1:
            return search_start + pos + 1

        return end

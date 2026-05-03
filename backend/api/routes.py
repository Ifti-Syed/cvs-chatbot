"""
API routes for CVS Chatbot.

Endpoints
---------
POST   /api/chat                         – RAG chat (query only, no PDF I/O)
POST   /api/upload-document              – Upload PDF → GCS → ingest into Qdrant
POST   /api/ingest-gcs-document          – Ingest a specific PDF already in GCS
POST   /api/ingest-all-gcs-documents     – Ingest every new PDF found in GCS
GET    /api/documents                    – List documents ingested into Qdrant
DELETE /api/documents/{document_name}    – Remove a document from Qdrant
GET    /api/gcs-documents                – List PDFs stored in GCS bucket
GET    /api/health                       – Health check
"""

import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Optional

from fastapi import APIRouter, HTTPException, UploadFile, File
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, Field

from backend.config import settings
from backend.ingestion.excel_processor import EXCEL_EXTENSIONS
from backend.ingestion.pipeline import IngestionPipeline
from backend.services.chat_service import ChatService
from backend.services.gcs_service import GCSService
from backend.services.openai_service import OpenAIService
from backend.services.vector_db import QdrantService

logger = logging.getLogger(__name__)

router = APIRouter()

# ---------------------------------------------------------------------------
# Module-level service singletons (lazy-initialised on first request)
# ---------------------------------------------------------------------------
_qdrant_service: Optional[QdrantService] = None
_openai_service: Optional[OpenAIService] = None
_chat_service: Optional[ChatService] = None
_gcs_service: Optional[GCSService] = None
_ingestion_pipeline: Optional[IngestionPipeline] = None


def _get_services() -> tuple[QdrantService, OpenAIService, ChatService, IngestionPipeline]:
    """Return (and lazily create) the core service singletons."""
    global _qdrant_service, _openai_service, _chat_service
    global _gcs_service, _ingestion_pipeline

    if _qdrant_service is None:
        _qdrant_service = QdrantService(settings)

    if _openai_service is None:
        _openai_service = OpenAIService(settings)

    if _chat_service is None:
        _chat_service = ChatService(_qdrant_service, _openai_service, settings)

    # GCS is optional – only initialise when bucket name is configured
    if _gcs_service is None and settings.gcs_bucket_name:
        try:
            _gcs_service = GCSService(settings)
        except Exception as exc:
            logger.warning("Could not initialise GCS service: %s", exc)

    if _ingestion_pipeline is None:
        _ingestion_pipeline = IngestionPipeline(
            qdrant_service=_qdrant_service,
            openai_service=_openai_service,
            config=settings,
            gcs_service=_gcs_service,
        )

    return _qdrant_service, _openai_service, _chat_service, _ingestion_pipeline


# ---------------------------------------------------------------------------
# Pydantic request / response models
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=4096)
    # Cap history to prevent oversized payloads; oldest turns are dropped client-side anyway
    conversation_history: list[dict[str, Any]] = Field(default_factory=list, max_length=50)
    stream: bool = Field(default=False)


class ChatResponse(BaseModel):
    answer: str
    sources: list[dict[str, Any]]


class DocumentInfo(BaseModel):
    document_name: str


class UploadResponse(BaseModel):
    status: str
    document_name: str
    total_chunks: int
    pages: int
    gcs_path: Optional[str] = None
    message: str


class GCSIngestRequest(BaseModel):
    """Request body for ingesting a specific GCS document."""
    blob_name: str = Field(
        ...,
        description="Full GCS blob path, e.g. 'sales_pdfs/product_catalog.pdf'",
    )
    document_name: Optional[str] = Field(
        default=None,
        description="Override logical name; defaults to filename without extension.",
    )
    force: bool = Field(
        default=False,
        description="Re-ingest even if the document already exists in Qdrant.",
    )


class BulkIngestResponse(BaseModel):
    status: str
    ingested: list[str]
    skipped: list[str]
    failed: list[dict[str, str]]
    message: str


class JsonTablesIngestRequest(BaseModel):
    subfolder: str = Field(
        default="Cf761-Tables",
        description="GCS subfolder under sales_pdfs/ containing the JSON files.",
    )
    document_name: Optional[str] = Field(
        default=None,
        description="Override logical document name (defaults to JSON filename stem).",
    )
    force: bool = Field(
        default=False,
        description="Re-ingest even if the document already exists in Qdrant.",
    )


class ImagesIngestRequest(BaseModel):
    subfolder: str = Field(
        default="CF761-Images",
        description="GCS subfolder under sales_pdfs/ containing the image files.",
    )
    document_name: str = Field(
        default="CF761",
        description="Logical document name for all images in this folder.",
    )
    force: bool = Field(
        default=False,
        description="Re-ingest even if the document already exists in Qdrant.",
    )


class DeleteResponse(BaseModel):
    status: str
    message: str


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@router.get("/health")
async def health_check() -> dict:
    return {"status": "healthy", "service": "CVS Chatbot API"}


# ---------------------------------------------------------------------------
# Chat
# ---------------------------------------------------------------------------

@router.post("/chat")
async def chat(
    request: ChatRequest,
):
    """
    RAG chat endpoint.
    - stream=false (default): returns JSON {answer, sources}
    - stream=true: streams tokens as Server-Sent Events
    """
    _, _, chat_service, _ = _get_services()

    if request.stream:
        try:
            generator = await chat_service.get_response_stream(
                message=request.message,
                conversation_history=request.conversation_history,
            )
        except RuntimeError as exc:
            logger.error("Stream retrieval error for message %.80r: %s", request.message, exc)
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        except Exception as exc:
            logger.exception("Stream setup error for message %.80r", request.message)
            raise HTTPException(status_code=500, detail="Failed to start stream. Please try again.") from exc

        return StreamingResponse(
            generator,
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    try:
        result = await chat_service.get_response(
            message=request.message,
            conversation_history=request.conversation_history,
        )
        return ChatResponse(answer=result["answer"], sources=result["sources"])
    except RuntimeError as exc:
        # RuntimeError is raised by _retrieve_and_rerank for known transient failures
        # (embedding generation, vector search). Pass the safe message through.
        logger.error("Chat retrieval error for message %.80r: %s", request.message, exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Unexpected chat error for message %.80r", request.message)
        raise HTTPException(status_code=500, detail="Chat request failed. Please try again.") from exc


# ---------------------------------------------------------------------------
# Upload document (HTTP multipart → GCS → Qdrant)
# ---------------------------------------------------------------------------

@router.post("/upload-document", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)) -> UploadResponse:
    """
    Accept a PDF or Excel upload, store it in GCS (if configured), then
    run the full ingestion pipeline to embed and index it in Qdrant.

    Supported: .pdf  |  .xlsx  .xls  .xlsm  .xlsb
    Duplicate documents are re-ingested (old vectors are deleted first).
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided.")

    original_name = Path(file.filename)
    suffix = original_name.suffix.lower()

    _PDF_EXT = {".pdf"}
    _SUPPORTED = _PDF_EXT | EXCEL_EXTENSIONS

    if suffix not in _SUPPORTED:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported file type '{suffix}'. "
                f"Accepted: {', '.join(sorted(_SUPPORTED))}"
            ),
        )

    is_pdf = suffix == ".pdf"
    document_name = original_name.stem
    qdrant_service, _, _, ingestion_pipeline = _get_services()

    file_bytes = await file.read()

    max_bytes = settings.max_upload_size_mb * 1024 * 1024
    if len(file_bytes) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum allowed size is {settings.max_upload_size_mb} MB.",
        )

    # Remove old vectors if the document was previously ingested
    try:
        if qdrant_service.check_document_exists(document_name):
            logger.info("'%s' already ingested; removing old vectors.", document_name)
            qdrant_service.delete_document(document_name)
    except Exception as exc:
        logger.warning("Could not check/delete existing document: %s", exc)

    # Upload to GCS (non-fatal if GCS is not configured)
    gcs_blob_name: Optional[str] = None
    if _gcs_service is not None:
        try:
            content_type = (
                "application/pdf" if is_pdf
                else "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
            gcs_blob_name = _gcs_service.upload_file(
                file_bytes, original_name.name, content_type=content_type
            )
        except Exception as exc:
            logger.warning(
                "GCS upload failed for '%s': %s — continuing without GCS.",
                original_name.name, exc,
            )

    # Write to a temp file for the ingestion pipeline
    tmp_path = ""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name

        source_folder = (
            f"gs://{settings.gcs_bucket_name}/{settings.gcs_folder_path}"
            if gcs_blob_name
            else "local_upload"
        )

        if is_pdf:
            result = await ingestion_pipeline.process_pdf(
                file_path=tmp_path,
                document_name=document_name,
                source_folder=source_folder,
            )
        else:
            result = await ingestion_pipeline.process_excel(
                file_path=tmp_path,
                document_name=document_name,
                source_folder=source_folder,
            )

    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Ingestion failed for '%s'", document_name)
        raise HTTPException(status_code=500, detail="Ingestion failed. Please try again.") from exc
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)

    file_type_label = "PDF" if is_pdf else "Excel"
    return UploadResponse(
        status="success",
        document_name=result["document_name"],
        total_chunks=result["total_chunks"],
        pages=result["pages"],
        gcs_path=gcs_blob_name,
        message=(
            f"[{file_type_label}] '{result['document_name']}' ingested: "
            f"{result['total_chunks']} chunks from {result['pages']} page(s)/sheet(s)."
            + (f" Stored at gs://{settings.gcs_bucket_name}/{gcs_blob_name}." if gcs_blob_name else "")
        ),
    )


# ---------------------------------------------------------------------------
# Image proxy — serves GCS-stored page images to the browser
# ---------------------------------------------------------------------------

@router.get("/images/{document_name}/{filename}")
async def get_image(document_name: str, filename: str):
    """
    Proxy endpoint that fetches a page image stored in GCS and returns it
    to the browser.  The blob path is:
        images/<document_name>/<filename>

    This avoids needing a public GCS bucket — all image requests flow
    through the API with the server's ADC credentials.
    """
    if _gcs_service is None:
        _get_services()  # trigger lazy init

    if _gcs_service is None:
        raise HTTPException(
            status_code=503,
            detail="Image storage (GCS) is not configured.",
        )

    blob_name = f"images/{document_name}/{filename}"

    try:
        image_bytes = _gcs_service.download_file(blob_name)
    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail=f"Image '{filename}' not found for document '{document_name}'.",
        )
    except Exception as exc:
        logger.exception("Could not fetch image '%s'", blob_name)
        raise HTTPException(status_code=500, detail="Failed to fetch image.") from exc

    # Detect content type from extension
    ext = Path(filename).suffix.lower().lstrip(".")
    content_type_map = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg", "webp": "image/webp"}
    content_type = content_type_map.get(ext, "image/png")

    return Response(content=image_bytes, media_type=content_type)


# ---------------------------------------------------------------------------
# Ingest a specific PDF already in GCS
# ---------------------------------------------------------------------------

@router.post("/ingest-gcs-document", response_model=UploadResponse)
async def ingest_gcs_document(body: GCSIngestRequest) -> UploadResponse:
    """
    Trigger ingestion of a PDF that is already stored in the GCS bucket.
    The PDF is NOT re-uploaded; it is only read during ingestion, then
    vectors are stored in Qdrant.
    """
    if _gcs_service is None:
        raise HTTPException(
            status_code=503,
            detail="GCS is not configured. Set GCS_BUCKET_NAME in your environment.",
        )

    qdrant_service, _, _, ingestion_pipeline = _get_services()

    # Derive document name
    filename = body.blob_name.split("/")[-1]
    document_name = (
        body.document_name
        or (filename[:-4] if filename.lower().endswith(".pdf") else filename)
    )

    # Duplicate check
    if not body.force:
        try:
            if qdrant_service.check_document_exists(document_name):
                raise HTTPException(
                    status_code=409,
                    detail=(
                        f"Document '{document_name}' is already indexed. "
                        "Pass force=true to re-ingest."
                    ),
                )
        except HTTPException:
            raise
        except Exception as exc:
            logger.warning("Duplicate check failed: %s", exc)
    else:
        # Remove existing vectors before re-ingesting
        try:
            if qdrant_service.check_document_exists(document_name):
                qdrant_service.delete_document(document_name)
        except Exception as exc:
            logger.warning("Could not delete existing vectors: %s", exc)

    try:
        result = await ingestion_pipeline.process_pdf_from_gcs(
            blob_name=body.blob_name,
            document_name=document_name,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("GCS ingestion failed for '%s'", body.blob_name)
        raise HTTPException(status_code=500, detail="Ingestion failed. Please try again.") from exc

    return UploadResponse(
        status="success",
        document_name=result["document_name"],
        total_chunks=result["total_chunks"],
        pages=result["pages"],
        gcs_path=body.blob_name,
        message=(
            f"'{result['document_name']}' ingested from GCS: "
            f"{result['total_chunks']} chunks from {result['pages']} page(s)."
        ),
    )


# ---------------------------------------------------------------------------
# Bulk-ingest all new PDFs from GCS
# ---------------------------------------------------------------------------

@router.post("/ingest-all-gcs-documents", response_model=BulkIngestResponse)
async def ingest_all_gcs_documents(force: bool = False) -> BulkIngestResponse:
    """
    List every PDF under the configured GCS folder and ingest any that
    are not yet indexed in Qdrant.

    Query params:
        force (bool) – if true, re-ingest documents that are already in Qdrant.
    """
    if _gcs_service is None:
        raise HTTPException(
            status_code=503,
            detail="GCS is not configured. Set GCS_BUCKET_NAME in your environment.",
        )

    qdrant_service, _, _, ingestion_pipeline = _get_services()

    all_docs = _gcs_service.list_documents()
    if not all_docs:
        return BulkIngestResponse(
            status="success",
            ingested=[],
            skipped=[],
            failed=[],
            message="No documents found in GCS bucket.",
        )

    ingested: list[str] = []
    skipped: list[str] = []
    failed: list[dict[str, str]] = []

    for doc_info in all_docs:
        blob_name    = doc_info["blob_name"]
        document_name = doc_info["document_name"]
        file_type    = doc_info["file_type"]  # "pdf" or "excel"

        # Skip already-indexed documents unless forced
        if not force:
            try:
                if qdrant_service.check_document_exists(document_name):
                    logger.info("Skipping '%s' – already in Qdrant.", document_name)
                    skipped.append(document_name)
                    continue
            except Exception as exc:
                logger.warning(
                    "Could not check existence of '%s': %s", document_name, exc
                )

        if force:
            try:
                if qdrant_service.check_document_exists(document_name):
                    qdrant_service.delete_document(document_name)
            except Exception as exc:
                logger.warning("Could not delete old vectors for '%s': %s", document_name, exc)

        try:
            if file_type == "excel":
                await ingestion_pipeline.process_excel_from_gcs(
                    blob_name=blob_name,
                    document_name=document_name,
                )
            else:
                await ingestion_pipeline.process_pdf_from_gcs(
                    blob_name=blob_name,
                    document_name=document_name,
                )
            ingested.append(document_name)
        except Exception as exc:
            logger.error("Failed to ingest '%s': %s", document_name, exc)
            failed.append({"document_name": document_name, "error": str(exc)})

    return BulkIngestResponse(
        status="success" if not failed else "partial",
        ingested=ingested,
        skipped=skipped,
        failed=failed,
        message=(
            f"Ingested {len(ingested)}, skipped {len(skipped)}, "
            f"failed {len(failed)} of {len(all_docs)} document(s)."
        ),
    )


# ---------------------------------------------------------------------------
# Reset collection and re-ingest everything from GCS
# ---------------------------------------------------------------------------

@router.post("/reset-and-reingest", response_model=BulkIngestResponse)
async def reset_and_reingest() -> BulkIngestResponse:
    """
    DESTRUCTIVE — drops the entire Qdrant collection, recreates it fresh,
    then re-ingests ALL data from structured JSON tables and captioned images.

    PDF content is intentionally excluded — all specifications come from the
    pre-structured JSON files for accuracy.

    Use this when:
      - JSON table files have been updated
      - New images have been added to CF761-Images/
      - Collection schema has changed and a full rebuild is needed
    """
    if _gcs_service is None:
        _get_services()
    if _gcs_service is None:
        raise HTTPException(
            status_code=503,
            detail="GCS is not configured. Set GCS_BUCKET_NAME in your environment.",
        )

    qdrant_service, _, _, ingestion_pipeline = _get_services()

    # Step 1 — wipe the collection
    logger.info("RESET: dropping and recreating Qdrant collection.")
    qdrant_service.reset_collection()
    logger.info("RESET: collection is empty and ready.")

    ingested: list[str] = []
    failed:   list[dict[str, str]] = []

    # Step 2 — re-ingest PDFs from Pdf files subfolder
    _PDF_SUBFOLDER = "Pdf files"
    try:
        pdf_files = _gcs_service.list_pdf_files(_PDF_SUBFOLDER)
    except Exception as exc:
        logger.error("Could not list PDF files from GCS: %s", exc)
        pdf_files = []

    for doc_info in pdf_files:
        doc_name  = doc_info["name_stem"]
        blob_name = doc_info["blob_name"]
        try:
            result = await ingestion_pipeline.process_pdf_from_gcs(
                blob_name=blob_name, document_name=doc_name,
            )
            ingested.append(doc_name)
            logger.info("Reset re-ingested PDF '%s': %d chunks.", doc_name, result["total_chunks"])
        except Exception as exc:
            logger.error("PDF re-ingest failed for '%s': %s", doc_name, exc)
            failed.append({"document_name": doc_name, "error": str(exc)})

    # Step 2b — re-ingest HR Policy PDFs (root-level HR-Policy/ subfolder)
    _HR_SUBFOLDER = "HR-Policy"
    try:
        hr_files = _gcs_service.list_pdfs_from_root_subfolder(_HR_SUBFOLDER)
    except Exception as exc:
        logger.error("Could not list HR PDF files from GCS: %s", exc)
        hr_files = []

    for doc_info in hr_files:
        doc_name  = doc_info["name_stem"]
        blob_name = doc_info["blob_name"]
        try:
            result = await ingestion_pipeline.process_pdf_from_gcs(
                blob_name=blob_name,
                document_name=doc_name,
                source_type_override="hr_policy",
            )
            ingested.append(doc_name)
            logger.info("Reset re-ingested HR PDF '%s': %d chunks.", doc_name, result["total_chunks"])
        except Exception as exc:
            logger.error("HR PDF re-ingest failed for '%s': %s", doc_name, exc)
            failed.append({"document_name": doc_name, "error": str(exc)})

    # Step 3 — re-ingest JSON tables
    _JSON_SUBFOLDER = "Cf761-Tables"
    try:
        json_files = _gcs_service.list_json_files(_JSON_SUBFOLDER)
    except Exception as exc:
        logger.error("Could not list JSON files: %s", exc)
        json_files = []

    for file_info in json_files:
        doc_name = file_info["name_stem"]
        try:
            result = await ingestion_pipeline.process_json_from_gcs(
                blob_name=file_info["blob_name"],
                document_name=doc_name,
            )
            ingested.append(doc_name)
            logger.info("Reset re-ingested JSON '%s': %d records.", doc_name, result["total_chunks"])
        except Exception as exc:
            logger.error("JSON re-ingest failed for '%s': %s", doc_name, exc)
            failed.append({"document_name": doc_name, "error": str(exc)})

    # Step 4 — re-ingest images
    _IMAGE_SUBFOLDER = "CF761-Images"
    _IMAGE_DOC_KEY   = "CF761-images"
    try:
        result = await ingestion_pipeline.process_images_from_gcs(
            subfolder=_IMAGE_SUBFOLDER,
            document_name=_IMAGE_DOC_KEY,
        )
        ingested.append(_IMAGE_DOC_KEY)
        logger.info("Reset re-ingested %d image caption(s).", result["total_chunks"])
    except Exception as exc:
        logger.error("Image re-ingest failed: %s", exc)
        failed.append({"document_name": _IMAGE_DOC_KEY, "error": str(exc)})

    status = "success" if not failed else "partial"
    return BulkIngestResponse(
        status=status,
        ingested=ingested,
        skipped=[],
        failed=failed,
        message=(
            f"Collection reset and rebuilt. "
            f"Ingested {len(ingested)} source(s), failed {len(failed)}."
        ),
    )


# ---------------------------------------------------------------------------
# Ingest JSON table files from GCS
# ---------------------------------------------------------------------------

@router.post("/ingest-json-tables", response_model=BulkIngestResponse)
async def ingest_json_tables(body: JsonTablesIngestRequest) -> BulkIngestResponse:
    """
    List every JSON file under sales_pdfs/<subfolder>/ in GCS and ingest each
    one as a set of JSON table records into Qdrant.

    Each record's 'text' field is embedded directly — no further chunking.
    All structured fields (table_number, duct_size, system, etc.) are stored
    in the Qdrant payload for precise filtering.
    """
    if _gcs_service is None:
        _get_services()
    if _gcs_service is None:
        raise HTTPException(
            status_code=503,
            detail="GCS is not configured. Set GCS_BUCKET_NAME in your environment.",
        )

    qdrant_service, _, _, ingestion_pipeline = _get_services()

    json_files = _gcs_service.list_json_files(body.subfolder)
    if not json_files:
        return BulkIngestResponse(
            status="success",
            ingested=[],
            skipped=[],
            failed=[],
            message=f"No JSON files found under sales_pdfs/{body.subfolder}/.",
        )

    ingested: list[str] = []
    skipped:  list[str] = []
    failed:   list[dict[str, str]] = []

    for file_info in json_files:
        blob_name     = file_info["blob_name"]
        document_name = body.document_name or file_info["name_stem"]

        if not body.force:
            try:
                if qdrant_service.check_document_exists(document_name):
                    logger.info("Skipping '%s' – already in Qdrant.", document_name)
                    skipped.append(document_name)
                    continue
            except Exception as exc:
                logger.warning("Existence check failed for '%s': %s", document_name, exc)

        if body.force:
            try:
                if qdrant_service.check_document_exists(document_name):
                    qdrant_service.delete_document(document_name)
            except Exception as exc:
                logger.warning("Could not delete old vectors for '%s': %s", document_name, exc)

        try:
            result = await ingestion_pipeline.process_json_from_gcs(
                blob_name=blob_name,
                document_name=document_name,
            )
            ingested.append(document_name)
            logger.info(
                "JSON ingested '%s': %d records.", document_name, result["total_chunks"]
            )
        except Exception as exc:
            logger.error("JSON ingestion failed for '%s': %s", blob_name, exc)
            failed.append({"document_name": document_name, "error": str(exc)})

    return BulkIngestResponse(
        status="success" if not failed else "partial",
        ingested=ingested,
        skipped=skipped,
        failed=failed,
        message=(
            f"JSON tables: ingested {len(ingested)}, "
            f"skipped {len(skipped)}, failed {len(failed)} "
            f"of {len(json_files)} file(s)."
        ),
    )


# ---------------------------------------------------------------------------
# Ingest images from GCS (captions via GPT-4o Vision)
# ---------------------------------------------------------------------------

@router.post("/ingest-images", response_model=UploadResponse)
async def ingest_images(body: ImagesIngestRequest) -> UploadResponse:
    """
    Download all images from sales_pdfs/<subfolder>/ in GCS, generate
    a detailed text caption for each using GPT-4o Vision, embed the
    captions, and upsert into Qdrant.

    The original images are also copied to images/<document_name>/ in GCS
    so the frontend /api/images proxy can display them alongside answers.
    """
    if _gcs_service is None:
        _get_services()
    if _gcs_service is None:
        raise HTTPException(
            status_code=503,
            detail="GCS is not configured. Set GCS_BUCKET_NAME in your environment.",
        )

    qdrant_service, _, _, ingestion_pipeline = _get_services()
    doc_key = f"{body.document_name}-images"

    if not body.force:
        try:
            if qdrant_service.check_document_exists(doc_key):
                raise HTTPException(
                    status_code=409,
                    detail=(
                        f"Images for '{body.document_name}' are already indexed. "
                        "Pass force=true to re-ingest."
                    ),
                )
        except HTTPException:
            raise
        except Exception as exc:
            logger.warning("Existence check failed: %s", exc)

    if body.force:
        try:
            if qdrant_service.check_document_exists(doc_key):
                qdrant_service.delete_document(doc_key)
        except Exception as exc:
            logger.warning("Could not delete old image vectors: %s", exc)

    try:
        result = await ingestion_pipeline.process_images_from_gcs(
            subfolder=body.subfolder,
            document_name=doc_key,
        )
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Image ingestion failed for subfolder '%s'", body.subfolder)
        raise HTTPException(status_code=500, detail="Image ingestion failed.") from exc

    return UploadResponse(
        status="success",
        document_name=result["document_name"],
        total_chunks=result["total_chunks"],
        pages=result["pages"],
        gcs_path=f"{settings.gcs_folder_path}/{body.subfolder}",
        message=(
            f"Captioned and indexed {result['total_chunks']} image(s) "
            f"from '{body.subfolder}' as '{result['document_name']}'."
        ),
    )


# ---------------------------------------------------------------------------
# List documents in Qdrant
# ---------------------------------------------------------------------------

@router.get("/documents", response_model=list[DocumentInfo])
async def list_documents() -> list[DocumentInfo]:
    """Return the names of all documents currently indexed in Qdrant."""
    qdrant_service, _, _, _ = _get_services()
    try:
        names = qdrant_service.get_documents()
        return [DocumentInfo(document_name=n) for n in names]
    except Exception as exc:
        logger.exception("Error listing documents")
        raise HTTPException(status_code=500, detail="Failed to list documents.") from exc


# ---------------------------------------------------------------------------
# Delete a document from Qdrant
# ---------------------------------------------------------------------------

@router.delete("/documents/{document_name}", response_model=DeleteResponse)
async def delete_document(document_name: str) -> DeleteResponse:
    """Delete all Qdrant vectors for the named document."""
    qdrant_service, _, _, _ = _get_services()
    try:
        if not qdrant_service.check_document_exists(document_name):
            raise HTTPException(
                status_code=404,
                detail=f"Document '{document_name}' not found in Qdrant.",
            )
        qdrant_service.delete_document(document_name)
        return DeleteResponse(
            status="success",
            message=f"Document '{document_name}' deleted from Qdrant.",
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Error deleting document '%s'", document_name)
        raise HTTPException(status_code=500, detail="Delete failed. Please try again.") from exc


# ---------------------------------------------------------------------------
# List PDFs in GCS bucket
# ---------------------------------------------------------------------------

@router.get("/gcs-documents")
async def list_gcs_documents() -> list[dict]:
    """
    Return metadata for every PDF stored in the GCS bucket folder.
    Useful for seeing what is available to ingest.
    """
    if _gcs_service is None:
        # Trigger lazy init so _gcs_service gets set if it can be
        _get_services()

    if _gcs_service is None:
        raise HTTPException(
            status_code=503,
            detail="GCS is not configured. Set GCS_BUCKET_NAME in your environment.",
        )

    try:
        return _gcs_service.list_documents()
    except Exception as exc:
        logger.exception("Error listing GCS documents")
        raise HTTPException(status_code=500, detail="GCS listing failed. Please try again.") from exc

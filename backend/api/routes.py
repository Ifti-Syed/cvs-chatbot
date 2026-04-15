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
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from backend.config import settings
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
    Accept a PDF upload, store it in GCS (if configured), then run the
    full ingestion pipeline to embed and index it in Qdrant.

    Duplicate documents are re-ingested (old vectors are deleted first).
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided.")

    original_name = Path(file.filename)
    if original_name.suffix.lower() != ".pdf":
        raise HTTPException(
            status_code=400,
            detail=f"Only PDF files are accepted. Got: '{original_name.suffix}'",
        )

    document_name = original_name.stem
    qdrant_service, _, _, ingestion_pipeline = _get_services()

    # Read file bytes once; reuse for GCS upload and local temp file
    pdf_bytes = await file.read()

    # Reject uploads that exceed the configured size limit
    max_bytes = settings.max_upload_size_mb * 1024 * 1024
    if len(pdf_bytes) > max_bytes:
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
            gcs_blob_name = _gcs_service.upload_pdf(pdf_bytes, original_name.name)
        except Exception as exc:
            logger.warning("GCS upload failed for '%s': %s – continuing without GCS.", original_name.name, exc)

    # Write to a temp file for the ingestion pipeline
    tmp_path = ""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(pdf_bytes)
            tmp_path = tmp.name

        # Derive source_folder from GCS path if available
        source_folder = (
            f"gs://{settings.gcs_bucket_name}/{settings.gcs_folder_path}"
            if gcs_blob_name
            else "local_upload"
        )

        result = await ingestion_pipeline.process_pdf(
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

    return UploadResponse(
        status="success",
        document_name=result["document_name"],
        total_chunks=result["total_chunks"],
        pages=result["pages"],
        gcs_path=gcs_blob_name,
        message=(
            f"'{result['document_name']}' ingested: "
            f"{result['total_chunks']} chunks from {result['pages']} page(s)."
            + (f" Stored at gs://{settings.gcs_bucket_name}/{gcs_blob_name}." if gcs_blob_name else "")
        ),
    )


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

    pdf_list = _gcs_service.list_pdfs()
    if not pdf_list:
        return BulkIngestResponse(
            status="success",
            ingested=[],
            skipped=[],
            failed=[],
            message="No PDFs found in GCS bucket.",
        )

    ingested: list[str] = []
    skipped: list[str] = []
    failed: list[dict[str, str]] = []

    for pdf_info in pdf_list:
        blob_name = pdf_info["blob_name"]
        document_name = pdf_info["document_name"]

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
            # Remove stale vectors before re-ingesting
            try:
                if qdrant_service.check_document_exists(document_name):
                    qdrant_service.delete_document(document_name)
            except Exception as exc:
                logger.warning("Could not delete old vectors for '%s': %s", document_name, exc)

        try:
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
            f"failed {len(failed)} of {len(pdf_list)} PDF(s)."
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
        return _gcs_service.list_pdfs()
    except Exception as exc:
        logger.exception("Error listing GCS documents")
        raise HTTPException(status_code=500, detail="GCS listing failed. Please try again.") from exc

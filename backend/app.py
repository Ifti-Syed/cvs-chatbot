"""
CVS Chatbot - FastAPI application factory.

Creates and configures the FastAPI app with CORS, API routes,
a startup/shutdown lifespan, and frontend static file serving.
"""

import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from backend.api.routes import router, _get_services
from backend.config import settings

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.DEBUG if not settings.is_production else logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lifespan — runs once at startup and once at shutdown
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    _log_startup_banner()
    _init_qdrant()

    poll_task = None
    if settings.gcs_bucket_name and settings.gcs_poll_interval > 0:
        # Run an immediate first-pass ingest, then keep polling
        poll_task = asyncio.create_task(_gcs_poll_loop())
        logger.info(
            "GCS auto-ingest enabled — polling every %ds.", settings.gcs_poll_interval
        )
    else:
        logger.info("GCS auto-ingest disabled (GCS_POLL_INTERVAL=0 or no bucket set).")

    yield

    # Shutdown
    if poll_task:
        poll_task.cancel()
        try:
            await poll_task
        except asyncio.CancelledError:
            pass
    logger.info("CVS Chatbot shutting down.")


async def _gcs_poll_loop() -> None:
    """Background task: ingest new GCS PDFs on startup, then re-check every interval."""
    # Small delay so the server is fully ready before the first ingest
    await asyncio.sleep(2)

    while True:
        await _ingest_new_gcs_pdfs()
        await asyncio.sleep(settings.gcs_poll_interval)


async def _ingest_new_gcs_pdfs() -> None:
    """Check GCS for PDFs not yet in Qdrant and ingest them."""
    try:
        qdrant_service, _, _, ingestion_pipeline = _get_services()

        # _get_services() lazily inits _gcs_service; import it after the call
        from backend.api.routes import _gcs_service
        if _gcs_service is None:
            return

        pdf_list = _gcs_service.list_pdfs()
        new_pdfs = [
            p for p in pdf_list
            if not qdrant_service.check_document_exists(p["document_name"])
        ]

        if not new_pdfs:
            logger.debug("GCS poll: no new PDFs found.")
            return

        logger.info("GCS poll: found %d new PDF(s) to ingest.", len(new_pdfs))
        for pdf in new_pdfs:
            try:
                result = await ingestion_pipeline.process_pdf_from_gcs(
                    blob_name=pdf["blob_name"],
                    document_name=pdf["document_name"],
                )
                logger.info(
                    "Auto-ingested '%s': %d chunks from %d page(s).",
                    result["document_name"],
                    result["total_chunks"],
                    result["pages"],
                )
            except Exception as exc:
                logger.error("Auto-ingest failed for '%s': %s", pdf["document_name"], exc)

    except Exception as exc:
        logger.error("GCS poll error: %s", exc)


def _log_startup_banner() -> None:
    gcs = (
        f"gs://{settings.gcs_bucket_name}/{settings.gcs_folder_path}"
        if settings.gcs_bucket_name
        else "not configured"
    )
    logger.info("=" * 60)
    logger.info("  CVS Chatbot")
    logger.info("  Environment : %s", settings.environment)
    logger.info("  Qdrant      : %s", settings.qdrant_url)
    logger.info("  Chat model  : %s", settings.openai_chat_model)
    logger.info("  Embed model : %s", settings.openai_embedding_model)
    logger.info("  GCS         : %s", gcs)
    logger.info("=" * 60)

    if not settings.openai_api_key:
        logger.warning("OPENAI_API_KEY is not set — chat endpoints will fail.")


def _init_qdrant() -> None:
    """Ensure the Qdrant collection exists before the first request.

    Raises SystemExit on failure so Cloud Run marks the instance as unhealthy
    and refuses to route traffic until a healthy instance is available.
    """
    if not settings.openai_api_key or not settings.openai_api_key.strip():
        logger.critical("OPENAI_API_KEY is not set. Cannot start — all chat requests would fail.")
        raise SystemExit(1)

    try:
        qdrant_service, _, _, _ = _get_services()
        qdrant_service.initialize_collection()
        logger.info("Qdrant collection '%s' ready.", settings.qdrant_collection_name)
    except Exception as exc:
        logger.critical("Qdrant init failed: %s — refusing to start.", exc)
        raise SystemExit(1) from exc


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------
def create_app() -> FastAPI:
    app = FastAPI(
        title="CVS Chatbot API",
        description="RAG-powered chatbot for Central Ventilation System",
        version="1.0.0",
        lifespan=lifespan,
        docs_url="/api/docs",
        redoc_url="/api/redoc",
        openapi_url="/api/openapi.json",
    )

    _add_cors(app)
    _add_routes(app)
    _add_frontend(app)

    return app


def _add_cors(app: FastAPI) -> None:
    # Parse comma-separated origins; strip whitespace from each entry.
    raw = settings.allowed_origins or "*"
    origins = [o.strip() for o in raw.split(",") if o.strip()]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )


def _add_routes(app: FastAPI) -> None:
    app.include_router(router, prefix="/api")

    @app.get("/health", tags=["health"])
    async def health():
        return {"status": "healthy", "service": "CVS Chatbot"}



def _add_frontend(app: FastAPI) -> None:
    frontend_dir = Path(__file__).parent.parent / "frontend"
    if not frontend_dir.exists():
        logger.warning("Frontend directory not found — serving API only.")
        return

    app.mount("/static", StaticFiles(directory=str(frontend_dir)), name="static")

    @app.get("/", include_in_schema=False)
    async def index():
        return FileResponse(str(frontend_dir / "index.html"))

    @app.get("/{full_path:path}", include_in_schema=False)
    async def spa_fallback(full_path: str):
        requested = frontend_dir / full_path
        if requested.is_file():
            return FileResponse(str(requested))
        return FileResponse(str(frontend_dir / "index.html"))


# ---------------------------------------------------------------------------
# Application instance (imported by uvicorn and main.py)
# ---------------------------------------------------------------------------
app = create_app()

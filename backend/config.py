"""
Configuration management for CVS Chatbot backend.
Uses pydantic-settings to load from environment variables.
"""

from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict

# Resolve .env relative to this file's location (backend/), then check project root too
_HERE = Path(__file__).parent          # backend/
_PROJECT_ROOT = _HERE.parent           # CVS_Chatbot/
_ENV_FILE = _PROJECT_ROOT / ".env" if (_PROJECT_ROOT / ".env").exists() else _HERE / ".env"


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=str(_ENV_FILE),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # OpenAI Settings — default empty so server starts without a key;
    # actual API calls will fail with a clear error if the key is not set.
    openai_api_key: str = ""

    # Qdrant Vector Database Settings
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: Optional[str] = None
    qdrant_collection_name: str = "cvs_documents"

    # Google Cloud Storage
    gcs_bucket_name: Optional[str] = None
    # Sub-folder inside the bucket where sales PDFs live (e.g. "sales_pdfs")
    gcs_folder_path: str = "sales_pdfs"

    # ── Model Settings ────────────────────────────────────────────────────────
    openai_embedding_model: str = "text-embedding-3-small"
    openai_chat_model: str = "gpt-4o"
    # Model used for LLM-based relevance selection.
    # This is NOT a trained cross-encoder reranker — it is a fast LLM that reads
    # the candidate chunks and picks the most relevant ones for the question.
    # A smaller/faster model is appropriate here; no generation quality is needed.
    openai_rerank_model: str = "gpt-4o-mini"

    # ── Chunking Settings ────────────────────────────────────────────────────
    chunk_size: int = 1000          # Target chunk character length
    chunk_overlap: int = 150       # Overlap between consecutive chunks
    min_chunk_size: int = 80       # Drop chunks shorter than this (orphan fragments)

    # ── Retrieval Settings ───────────────────────────────────────────────────
    #
    # Retrieval is a two-stage funnel:
    #
    #   Stage 1 — Candidate pool (top_k_results):
    #     Hybrid search (dense + keyword) retrieves this many chunks from Qdrant.
    #     Query expansion adds more.  The candidate pool is wider than what the
    #     answer LLM will see, giving the relevance selector material to work with.
    #
    #   Stage 2 — LLM relevance selection (rerank_top_k):
    #     The LLM selects the best rerank_top_k chunks from the candidate pool.
    #     ONLY these final chunks are passed to the answer LLM as context.
    #     The answer LLM never sees the full candidate pool.
    #
    top_k_results: int = 12  # Stage 1: candidate pool per search (pre-selection)
    rerank_top_k: int = 4     # Stage 2: final chunks sent to answer LLM
                              #   4 chunks keeps context tight; reduces dilution
                              #   and reranker prompt size without losing coverage.

    # Minimum cosine similarity score to accept a dense-search hit.
    # 0.30 is intentionally permissive: the reranker handles final quality
    # filtering, so it is better to include borderline chunks here than to
    # drop them before the reranker ever sees them.
    # Keyword hits always bypass this threshold (they matched on text, not vectors).
    # Raise toward 0.40 only if you consistently see clearly off-topic chunks
    # reaching the final answer despite reranking.
    min_score_threshold: float = 0.30

    # ── Reranker reliability ─────────────────────────────────────────────────
    # Timeout (seconds) for a single reranker API call.
    rerank_timeout_seconds: int = 12
    # How many additional attempts to make after the first failure.
    # Total attempts = rerank_max_retries + 1.
    rerank_max_retries: int = 2

    # ── Feature Flags ────────────────────────────────────────────────────────
    # Enable LLM-based reranking of retrieved chunks
    enable_reranking: bool = True
    # Enable hybrid search (dense + full-text keyword) with RRF fusion
    enable_hybrid_search: bool = True

    # ── Generation Settings ──────────────────────────────────────────────────
    max_tokens: int = 1200
    # Seed for OpenAI completions — improves answer reproducibility.
    # Same seed + same prompt → same output within a model version.
    # Does NOT guarantee identical results across model updates.
    openai_seed: int = 42

    # GCS polling — how often to check for new PDFs (seconds). 0 = disabled.
    gcs_poll_interval: int = 300  # 5 minutes default

    # ── Security Settings ────────────────────────────────────────────────────
    # Comma-separated list of allowed CORS origins.
    # Use "*" to allow all origins (acceptable when no credentials are used).
    # Example: "https://myapp.run.app,https://staging.myapp.run.app"
    allowed_origins: str = "*"

    # Maximum PDF upload size in megabytes.
    max_upload_size_mb: int = 50

    # Application Settings
    environment: str = "development"
    port: int = 8000

    @property
    def is_production(self) -> bool:
        return self.environment.lower() == "production"

    @property
    def qdrant_is_cloud(self) -> bool:
        return self.qdrant_api_key is not None and self.qdrant_api_key.strip() != ""


def get_settings() -> Settings:
    """Factory function to create settings instance."""
    return Settings()


# Global settings instance
settings = get_settings()

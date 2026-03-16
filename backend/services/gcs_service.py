"""
Google Cloud Storage service for CVS Chatbot.

Handles PDF upload, download, and listing from the GCS bucket.
PDFs are stored at: gs://<bucket>/sales_pdfs/<filename>.pdf

Authentication uses Application Default Credentials (ADC):
  - Locally:  run `gcloud auth application-default login`
  - Cloud Run: the service account attached to the instance is used automatically
"""

import io
import logging
from datetime import datetime, timezone
from typing import Optional

from google.cloud import storage
from google.cloud.exceptions import NotFound

from backend.config import Settings

logger = logging.getLogger(__name__)


class GCSService:
    """Service for interacting with Google Cloud Storage."""

    def __init__(self, config: Settings):
        if not config.gcs_bucket_name:
            raise ValueError(
                "GCS_BUCKET_NAME must be set in your environment to use GCS features."
            )

        self.bucket_name: str = config.gcs_bucket_name
        self.folder_path: str = config.gcs_folder_path.strip("/")  # e.g. "sales_pdfs"

        # storage.Client() picks up credentials via ADC automatically
        self._client = storage.Client()
        self._bucket = self._client.bucket(self.bucket_name)

        logger.info(
            "GCS service ready. Bucket: '%s', Folder: '%s'",
            self.bucket_name,
            self.folder_path,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def upload_pdf(self, file_bytes: bytes, filename: str) -> str:
        """
        Upload a PDF file to GCS under the configured folder path.

        Args:
            file_bytes: Raw bytes of the PDF.
            filename:   Bare filename including extension (e.g. "product_catalog.pdf").

        Returns:
            The GCS blob name (relative path), e.g. "sales_pdfs/product_catalog.pdf".
        """
        blob_name = self._make_blob_name(filename)
        blob = self._bucket.blob(blob_name)

        blob.upload_from_string(
            file_bytes,
            content_type="application/pdf",
            # Set custom metadata so we can track when it was uploaded
            num_retries=3,
        )
        blob.metadata = {"uploaded_at": datetime.now(timezone.utc).isoformat()}
        blob.patch()

        logger.info(
            "Uploaded '%s' to gs://%s/%s (%d bytes)",
            filename,
            self.bucket_name,
            blob_name,
            len(file_bytes),
        )
        return blob_name

    def download_pdf(self, blob_name: str) -> bytes:
        """
        Download a PDF from GCS by its blob name.

        Args:
            blob_name: Full GCS object path (e.g., "sales_pdfs/document.pdf").

        Returns:
            Raw PDF bytes.

        Raises:
            FileNotFoundError: If the blob does not exist.
        """
        blob = self._bucket.blob(blob_name)
        try:
            data = blob.download_as_bytes()
            logger.info(
                "Downloaded gs://%s/%s (%d bytes)",
                self.bucket_name,
                blob_name,
                len(data),
            )
            return data
        except NotFound:
            raise FileNotFoundError(
                f"File not found in GCS: gs://{self.bucket_name}/{blob_name}"
            )

    def list_pdfs(self) -> list[dict]:
        """
        List all PDF files stored under the configured folder path.

        Returns:
            List of dicts, each containing:
                blob_name    – full GCS object path
                filename     – bare filename
                document_name – filename without .pdf extension
                size_bytes   – file size
                updated      – ISO-8601 last-modified timestamp
                gcs_uri      – gs:// URI
        """
        prefix = f"{self.folder_path}/" if self.folder_path else ""
        blobs = self._client.list_blobs(self.bucket_name, prefix=prefix)

        results: list[dict] = []
        for blob in blobs:
            # Skip the "folder" placeholder object itself
            if blob.name.endswith("/"):
                continue
            if not blob.name.lower().endswith(".pdf"):
                continue

            filename = blob.name.split("/")[-1]
            document_name = filename[:-4] if filename.lower().endswith(".pdf") else filename
            results.append(
                {
                    "blob_name": blob.name,
                    "filename": filename,
                    "document_name": document_name,
                    "size_bytes": blob.size,
                    "updated": blob.updated.isoformat() if blob.updated else None,
                    "gcs_uri": f"gs://{self.bucket_name}/{blob.name}",
                }
            )

        logger.info(
            "Listed %d PDF(s) under gs://%s/%s",
            len(results),
            self.bucket_name,
            prefix,
        )
        return results

    def check_file_exists(self, blob_name: str) -> bool:
        """Return True if the given blob exists in the bucket."""
        return self._bucket.blob(blob_name).exists()

    def get_upload_timestamp(self, blob_name: str) -> Optional[str]:
        """
        Return the ISO-8601 upload timestamp for a blob, or None if not found.

        Prefers the custom 'uploaded_at' metadata field; falls back to
        the GCS 'updated' (last-modified) time.
        """
        blob = self._bucket.blob(blob_name)
        try:
            blob.reload()
            # Custom metadata takes priority
            if blob.metadata and blob.metadata.get("uploaded_at"):
                return blob.metadata["uploaded_at"]
            # Fall back to GCS last-modified time
            if blob.updated:
                return blob.updated.isoformat()
            return None
        except NotFound:
            return None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _make_blob_name(self, filename: str) -> str:
        """Build the full GCS blob path for a given filename."""
        return f"{self.folder_path}/{filename}" if self.folder_path else filename

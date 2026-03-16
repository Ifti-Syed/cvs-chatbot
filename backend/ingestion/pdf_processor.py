"""
PDF text extraction for CVS Chatbot ingestion pipeline.
Uses pypdf to extract per-page text content.
"""

import logging
from pathlib import Path

from pypdf import PdfReader

logger = logging.getLogger(__name__)


class PDFProcessor:
    """Extracts text content from PDF files on a per-page basis."""

    def extract_text(self, file_path: str) -> list[dict]:
        """
        Extract text from all pages of a PDF file.

        Args:
            file_path: Absolute path to the PDF file.

        Returns:
            List of dicts, each with:
                - page (int): 1-based page number
                - text (str): extracted text for that page
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"PDF file not found: {file_path}")
        if not path.suffix.lower() == ".pdf":
            raise ValueError(f"Expected a PDF file, got: {path.suffix}")

        logger.info("Extracting text from '%s'", path.name)
        reader = PdfReader(str(path))
        pages: list[dict] = []

        for page_index, page in enumerate(reader.pages, start=1):
            try:
                text = page.extract_text() or ""
                text = self._clean_text(text)
            except Exception as exc:
                logger.warning(
                    "Could not extract text from page %d: %s", page_index, exc
                )
                text = ""

            if text.strip():
                pages.append({"page": page_index, "text": text})
            else:
                logger.debug("Page %d has no extractable text, skipping.", page_index)

        logger.info(
            "Extracted text from %d/%d pages in '%s'.",
            len(pages),
            len(reader.pages),
            path.name,
        )
        return pages

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _clean_text(text: str) -> str:
        """Basic text cleanup: normalize whitespace, remove control characters."""
        import re

        # Replace multiple whitespace (except newlines) with a single space
        text = re.sub(r"[ \t]+", " ", text)
        # Collapse 3+ consecutive newlines into two
        text = re.sub(r"\n{3,}", "\n\n", text)
        # Remove null bytes and other common control characters
        text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
        return text.strip()

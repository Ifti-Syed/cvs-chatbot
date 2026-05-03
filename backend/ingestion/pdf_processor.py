"""
PDF text and table extraction for CVS Chatbot ingestion pipeline.

Uses pdfplumber instead of pypdf because pdfplumber:
  - Understands PDF layout and reading order.
  - Detects table boundaries via PDF graphics operators.
  - Extracts table cells as structured rows, not garbled whitespace.

Table rendering strategy:
  Each table is converted to pipe-delimited Markdown so column alignment
  is preserved through the chunking and embedding steps.  The LLM can
  therefore read "| EI 120 | Circular | 2 hours |" and understand it as
  a structured row, not a run-on fragment of noise.
"""

import logging
import re
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class PDFProcessor:
    """
    Extracts per-page text content (including formatted tables) from PDF files.

    Public API is identical to the old pypdf version so the ingestion
    pipeline requires no changes to call sites.
    """

    def extract_text(self, file_path: str) -> list[dict]:
        """
        Extract text from all pages of a PDF file.

        Returns:
            List of dicts — one per non-empty page:
              { "page": int (1-based), "text": str }
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"PDF file not found: {file_path}")
        if path.suffix.lower() != ".pdf":
            raise ValueError(f"Expected a .pdf file, got: {path.suffix}")

        try:
            import pdfplumber
        except ImportError as exc:
            raise ImportError(
                "pdfplumber is required for PDF processing. "
                "Run: pip install pdfplumber"
            ) from exc

        logger.info("Extracting text from '%s'", path.name)
        pages: list[dict] = []

        with pdfplumber.open(str(path)) as pdf:
            total = len(pdf.pages)
            for pdf_page in pdf.pages:
                page_num = pdf_page.page_number  # 1-based
                try:
                    text = self._extract_page_content(pdf_page)
                    text = self._clean_text(text)
                except Exception as exc:
                    logger.warning("Could not extract page %d: %s", page_num, exc)
                    text = ""

                if text.strip():
                    pages.append({"page": page_num, "text": text})
                else:
                    logger.debug("Page %d: no extractable content, skipping.", page_num)

        logger.info(
            "Extracted text from %d/%d pages in '%s'.",
            len(pages), total, path.name,
        )
        return pages

    # ------------------------------------------------------------------
    # Per-page extraction
    # ------------------------------------------------------------------

    def _extract_page_content(self, page) -> str:
        """
        Combine general layout text with pipe-delimited table blocks.

        Strategy:
          1. Extract general text (preserves reading order).
          2. Extract each detected table as a pipe-delimited block.
          3. Append the structured table blocks after the general text.

        The structured version is appended rather than substituted so we
        don't lose text that appears between or around the table cells.
        Table text that already appears garbled in the general extraction
        is acceptable — the clean structured block that follows gives the
        embedding model the signal it needs.
        """
        general_text = page.extract_text(x_tolerance=3, y_tolerance=3) or ""

        raw_tables = page.extract_tables()
        table_blocks: list[str] = []
        for raw_table in (raw_tables or []):
            if raw_table:
                block = self._table_to_text(raw_table)
                if block:
                    table_blocks.append(block)

        if not table_blocks:
            return general_text

        parts = [general_text] if general_text.strip() else []
        parts.extend(table_blocks)
        return "\n\n".join(parts)

    # ------------------------------------------------------------------
    # Table rendering
    # ------------------------------------------------------------------

    @staticmethod
    def _table_to_text(rows: list[list[Optional[str]]]) -> str:
        """
        Render a pdfplumber table as aligned pipe-delimited Markdown.

        Input : [ ["Header A", "Header B"], ["val 1", "val 2"], ... ]
        Output:
            | Header A | Header B |
            |----------|----------|
            | val 1    | val 2    |
        """
        if not rows:
            return ""

        # Normalise cells
        cleaned: list[list[str]] = [
            [str(cell).strip() if cell is not None else "" for cell in row]
            for row in rows
        ]

        # Drop completely blank tables
        if all(cell == "" for row in cleaned for cell in row):
            return ""

        col_count = max(len(row) for row in cleaned) if cleaned else 0
        if col_count == 0:
            return ""

        # Calculate column widths (minimum 1 so separator is never empty)
        widths = [1] * col_count
        for row in cleaned:
            for i, cell in enumerate(row):
                if i < col_count:
                    widths[i] = max(widths[i], len(cell))

        def fmt(row: list[str]) -> str:
            cells = [
                (row[i] if i < len(row) else "").ljust(widths[i])
                for i in range(col_count)
            ]
            return "| " + " | ".join(cells) + " |"

        lines = [fmt(cleaned[0])]
        lines.append("| " + " | ".join("-" * widths[i] for i in range(col_count)) + " |")
        for row in cleaned[1:]:
            lines.append(fmt(row))

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Text cleanup
    # ------------------------------------------------------------------

    @staticmethod
    def _clean_text(text: str) -> str:
        """Normalize whitespace and strip control characters."""
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
        return text.strip()

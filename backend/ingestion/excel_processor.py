"""
Excel ingestion for CVS Chatbot.

Each sheet is split into row-groups of ROWS_PER_CHUNK rows.
Every chunk starts with the sheet name + full column headers so the LLM
always knows what each cell value means — no matter which chunk is retrieved.

Example chunk:
    Sheet: Fire Ratings (rows 1-40 of 120)

    | Duct Ref  | Type         | Rating | Duration |
    |-----------|--------------|--------|----------|
    | MSTM240   | Three Sided  | EI 120 | 2 hours  |
    | MSTM241   | Four Sided   | EI 60  | 1 hour   |
    ...
"""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

EXCEL_EXTENSIONS = {".xlsx", ".xls", ".xlsm", ".xlsb"}

# Number of data rows per chunk (header is always prepended, so this is
# the rows below the header line).  40 rows × ~80 chars/row ≈ 3 200 chars,
# which fits comfortably within a 4 096-char embedding window.
ROWS_PER_CHUNK = 40


class ExcelProcessor:
    """
    Reads Excel workbooks and returns pre-chunked text in the same format
    as PDFProcessor: [{"page": int, "text": str, "sheet_name": str}]

    Key difference from a naive implementation: each returned dict is already
    a self-contained chunk with headers.  The pipeline's character-based
    chunker will leave them intact (they are well under chunk_size).
    """

    def extract_text(self, file_path: str) -> list[dict]:
        """
        Extract and pre-chunk all sheets of an Excel file.

        Returns:
            List of dicts — one per row-group:
              { "page": int (global, 1-based), "text": str, "sheet_name": str }
        """
        try:
            import pandas as pd
        except ImportError as exc:
            raise ImportError(
                "pandas and openpyxl are required. Run: pip install pandas openpyxl"
            ) from exc

        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Excel file not found: {file_path}")
        if path.suffix.lower() not in EXCEL_EXTENSIONS:
            raise ValueError(
                f"Expected an Excel file ({', '.join(EXCEL_EXTENSIONS)}), got: {path.suffix}"
            )

        logger.info("Extracting text from Excel '%s'", path.name)

        try:
            # dtype=str: keeps "120" as "120" not "120.0"
            sheets: dict = pd.read_excel(
                str(path), sheet_name=None, header=0, dtype=str
            )
        except Exception as exc:
            raise ValueError(f"Could not open '{path.name}': {exc}") from exc

        pages: list[dict] = []
        global_page = 1  # each chunk gets its own page number

        for sheet_name, df in sheets.items():
            sheet_chunks = self._sheet_to_chunks(str(sheet_name), df)
            n_chunks = len(sheet_chunks)

            for chunk_idx, chunk_text in enumerate(sheet_chunks):
                if not chunk_text.strip():
                    continue
                pages.append({
                    "page":       global_page,
                    "text":       chunk_text,
                    "sheet_name": str(sheet_name),
                })
                global_page += 1

            if n_chunks:
                logger.info(
                    "Sheet '%s': %d row(s) → %d chunk(s).",
                    sheet_name, len(df), n_chunks,
                )
            else:
                logger.debug("Sheet '%s' produced no chunks, skipping.", sheet_name)

        logger.info(
            "Excel '%s': %d sheet(s) → %d chunk(s) total.",
            path.name, len(sheets), len(pages),
        )
        return pages

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _sheet_to_chunks(self, sheet_name: str, df) -> list[str]:
        """
        Split one DataFrame sheet into row-grouped chunks.
        Every chunk includes the sheet name and full column headers.
        """
        try:
            import pandas as pd
        except ImportError:
            return []

        # Drop fully-empty rows and columns
        df = df.dropna(how="all").dropna(axis=1, how="all").fillna("")

        if df.empty:
            return []

        headers = [str(col).strip() for col in df.columns]
        col_count = len(headers)
        if col_count == 0:
            return []

        # Column widths — at least as wide as the header
        widths = [max(len(h), 1) for h in headers]
        for _, row in df.iterrows():
            for i, val in enumerate(row):
                if i < col_count:
                    widths[i] = max(widths[i], len(str(val).strip()))

        def fmt(values: list[str]) -> str:
            cells = [str(v).strip().ljust(widths[i]) for i, v in enumerate(values)]
            return "| " + " | ".join(cells) + " |"

        header_row = fmt(headers)
        sep_row    = "| " + " | ".join("-" * w for w in widths) + " |"

        total_rows = len(df)
        chunks: list[str] = []

        for start in range(0, max(total_rows, 1), ROWS_PER_CHUNK):
            end = min(start + ROWS_PER_CHUNK, total_rows)
            chunk_df = df.iloc[start:end]

            # Row range label so the LLM knows where in the sheet this is
            if total_rows > ROWS_PER_CHUNK:
                range_label = f"rows {start + 1}–{end} of {total_rows}"
            else:
                range_label = f"{total_rows} row(s)"

            lines = [
                f"Sheet: {sheet_name} ({range_label})",
                "",
                header_row,
                sep_row,
            ]
            for _, row in chunk_df.iterrows():
                lines.append(fmt([str(v) for v in row]))

            chunks.append("\n".join(lines))

        return chunks

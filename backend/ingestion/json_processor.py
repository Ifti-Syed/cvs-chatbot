"""
JSON table record ingestion for CVS Chatbot.

Each JSON file is a list of pre-extracted table rows.  Every record contains:
  - text     : natural-language description of the row — used directly as the
               embedding input.  One record = one Qdrant point.
  - metadata : flat dict of structured values (stored in Qdrant payload for
               precise filtering).
  - top-level context fields: table_number, title, system, insulation, duct_size,
               file (source Excel filename).

No further chunking is applied because each text field is already a complete,
self-contained sentence (~200-400 chars) and splitting would destroy the
row-level context that makes these records so precise.
"""

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class JSONTableProcessor:
    """Converts JSON table files (list of row records) into pipeline-compatible page dicts."""

    def load_file(self, file_path: str) -> list[dict]:
        """Load and validate a JSON table file.

        Returns:
            List of raw record dicts as stored in the file.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"JSON file not found: {file_path}")
        if path.suffix.lower() != ".json":
            raise ValueError(f"Expected a .json file, got: {path.suffix}")

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, list):
            raise ValueError(
                f"Expected a JSON array in '{path.name}', got {type(data).__name__}"
            )

        logger.info("Loaded %d records from '%s'.", len(data), path.name)
        return data

    def records_to_pages(
        self, records: list[dict], source_filename: str
    ) -> list[dict]:
        """Convert JSON records to the pipeline's standard page-dict format.

        Each record with a non-empty 'text' field becomes one page dict.
        Extra structured fields are carried through to the Qdrant payload.

        Returns:
            List of dicts containing at minimum: page (int), text (str),
            plus all JSON-specific fields.
        """
        pages: list[dict] = []
        for i, record in enumerate(records):
            text = (record.get("text") or "").strip()
            if not text:
                logger.debug(
                    "Skipping record %d in '%s' — empty text field.", i, source_filename
                )
                continue

            pages.append({
                "page":              i + 1,
                "text":              text,
                "table_number":      str(record.get("table_number") or ""),
                "table_description": str(record.get("table_description") or ""),
                "title":             str(record.get("title") or ""),
                "system":            str(record.get("system") or ""),
                "insulation":        str(record.get("insulation") or ""),
                "duct_size":         str(record.get("duct_size") or ""),
                "source_file":       str(record.get("file") or source_filename),
                "json_metadata":     record.get("metadata") or {},
            })

        logger.info(
            "Converted %d/%d records from '%s' into page dicts.",
            len(pages), len(records), source_filename,
        )
        return pages

"""
Qdrant vector database service for CVS Chatbot.
Handles all vector storage and retrieval operations.
"""

import logging
import uuid
from typing import Optional

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
    FilterSelector,
    PayloadSchemaType,
)

from backend.config import Settings

logger = logging.getLogger(__name__)

# Vector size for text-embedding-3-small
EMBEDDING_VECTOR_SIZE = 1536


class QdrantService:
    """Service for interacting with Qdrant vector database."""

    def __init__(self, config: Settings):
        self.config = config
        self.collection_name = config.qdrant_collection_name

        if config.qdrant_is_cloud:
            # Qdrant Cloud (remote with API key)
            logger.info("Connecting to Qdrant Cloud at %s", config.qdrant_url)
            self.client = QdrantClient(
                url=config.qdrant_url,
                api_key=config.qdrant_api_key,
            )
        elif config.qdrant_url in (":memory:", "memory"):
            # In-memory mode — data is lost on restart, good for quick testing
            logger.info("Using Qdrant in-memory mode (data is not persisted).")
            self.client = QdrantClient(":memory:")
        elif config.qdrant_url.startswith("local://") or config.qdrant_url == "local":
            # Local file storage — persists data in ./qdrant_data, NO Docker needed
            path = config.qdrant_url.replace("local://", "") or "./qdrant_data"
            logger.info("Using Qdrant local storage at '%s' (no Docker required).", path)
            self.client = QdrantClient(path=path)
        else:
            # Remote Qdrant server (e.g. http://localhost:6333 via Docker)
            logger.info("Connecting to Qdrant server at %s", config.qdrant_url)
            self.client = QdrantClient(url=config.qdrant_url)

    def initialize_collection(self) -> None:
        """Create the Qdrant collection and payload index if they do not already exist."""
        existing = [c.name for c in self.client.get_collections().collections]

        if self.collection_name not in existing:
            logger.info("Creating collection '%s'.", self.collection_name)
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=EMBEDDING_VECTOR_SIZE,
                    distance=Distance.COSINE,
                ),
            )
            logger.info("Collection '%s' created successfully.", self.collection_name)
        else:
            logger.info("Collection '%s' already exists.", self.collection_name)

        # Ensure a keyword index exists on document_name so filter queries work.
        # This call is idempotent — safe to run on existing collections.
        try:
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="document_name",
                field_schema=PayloadSchemaType.KEYWORD,
            )
            logger.info("Payload index on 'document_name' ready.")
        except Exception as exc:
            # Index may already exist — not an error
            logger.debug("Payload index creation skipped: %s", exc)

    def upsert_vectors(self, vectors: list[dict]) -> None:
        """
        Batch upsert vectors into the collection.

        Each item in `vectors` must have:
            - id (str | None): optional, will be auto-generated if missing
            - vector (list[float]): embedding
            - payload (dict): metadata (document_name, page_number, text_chunk, chunk_index)
        """
        points = []
        for item in vectors:
            point_id = item.get("id") or str(uuid.uuid4())
            points.append(
                PointStruct(
                    id=point_id,
                    vector=item["vector"],
                    payload=item["payload"],
                )
            )

        self.client.upsert(
            collection_name=self.collection_name,
            points=points,
            wait=True,
        )
        logger.info("Upserted %d vectors into '%s'.", len(points), self.collection_name)

    def search(self, query_vector: list[float], top_k: int) -> list[dict]:
        """
        Perform semantic similarity search.

        Returns a list of dicts with keys:
            text_chunk, document_name, page_number, score
        """
        response = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=top_k,
            with_payload=True,
        )

        chunks = []
        for hit in response.points:
            payload = hit.payload or {}
            chunks.append(
                {
                    "text_chunk": payload.get("text_chunk", ""),
                    "document_name": payload.get("document_name", "unknown"),
                    "page_number": payload.get("page_number", 0),
                    "chunk_index": payload.get("chunk_index", 0),
                    "score": round(float(hit.score), 4),
                }
            )
        return chunks

    def get_documents(self) -> list[str]:
        """Return a list of unique document names stored in the collection."""
        try:
            # Scroll through all points to collect unique document names
            document_names: set[str] = set()
            offset = None

            while True:
                records, offset = self.client.scroll(
                    collection_name=self.collection_name,
                    limit=250,
                    offset=offset,
                    with_payload=["document_name"],
                    with_vectors=False,
                )
                for record in records:
                    if record.payload:
                        name = record.payload.get("document_name")
                        if name:
                            document_names.add(name)
                if offset is None:
                    break

            return sorted(document_names)
        except Exception as exc:
            logger.error("Error fetching documents: %s", exc)
            return []

    def delete_document(self, document_name: str) -> None:
        """Delete all vectors associated with a specific document name."""
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=FilterSelector(
                filter=Filter(
                    must=[
                        FieldCondition(
                            key="document_name",
                            match=MatchValue(value=document_name),
                        )
                    ]
                )
            ),
            wait=True,
        )
        logger.info("Deleted all vectors for document '%s'.", document_name)

    def check_document_exists(self, document_name: str) -> bool:
        """Return True if at least one vector for the given document exists."""
        results = self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter=Filter(
                must=[
                    FieldCondition(
                        key="document_name",
                        match=MatchValue(value=document_name),
                    )
                ]
            ),
            limit=1,
            with_payload=False,
            with_vectors=False,
        )
        records, _ = results
        return len(records) > 0

"""
Qdrant vector database service.
Handles dense search, full-text keyword search, and hybrid RRF fusion.
"""
from httpx import Timeout
import logging
import uuid
from typing import Optional

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    HnswConfigDiff,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
    MatchText,
    FilterSelector,
    PayloadSchemaType,
    TextIndexParams,
    TokenizerType,
)

from backend.config import Settings

logger = logging.getLogger(__name__)

# Vector size for text-embedding-3-small
EMBEDDING_VECTOR_SIZE = 1536

# RRF constant — standard value, higher = less penalty for lower ranks
_RRF_K = 60


class QdrantService:
    """Service for interacting with Qdrant vector database."""

    def __init__(self, config: Settings):
        self.config = config
        self.collection_name = config.qdrant_collection_name

        if config.qdrant_is_cloud:
            logger.info("Connecting to Qdrant Cloud at %s", config.qdrant_url)
            self.client = QdrantClient(
                url=config.qdrant_url,
                api_key=config.qdrant_api_key,
                timeout=30.0,
            )
        elif config.qdrant_url in (":memory:", "memory"):
            logger.info("Using Qdrant in-memory mode (data is not persisted).")
            self.client = QdrantClient(":memory:")
        elif config.qdrant_url.startswith("local://") or config.qdrant_url == "local":
            path = config.qdrant_url.replace("local://", "") or "./qdrant_data"
            logger.info("Using Qdrant local storage at '%s'.", path)
            self.client = QdrantClient(path=path)
        else:
            logger.info("Connecting to Qdrant server at %s", config.qdrant_url)
            self.client = QdrantClient(url=config.qdrant_url)

    # ------------------------------------------------------------------
    # Collection management
    # ------------------------------------------------------------------

    def initialize_collection(self) -> None:
        """Create the Qdrant collection and all payload indexes."""
        existing = [c.name for c in self.client.get_collections().collections]

        if self.collection_name not in existing:
            logger.info("Creating collection '%s'.", self.collection_name)
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config={
                    "dense_vector": VectorParams(
                        size=EMBEDDING_VECTOR_SIZE,
                        distance=Distance.COSINE,
                        hnsw_config=HnswConfigDiff(m=24, ef_construct=256),
                    )
                },
            )
            logger.info("Collection '%s' created.", self.collection_name)
        else:
            logger.info("Collection '%s' already exists.", self.collection_name)

        # Keyword index on document_name (for filtering by document)
        self._ensure_payload_index(
            field_name="document_name",
            field_schema=PayloadSchemaType.KEYWORD,
        )

        # Full-text index on text_chunk (enables keyword/hybrid search)
        self._ensure_text_index(field_name="text_chunk")

    def _ensure_payload_index(self, field_name: str, field_schema) -> None:
        try:
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name=field_name,
                field_schema=field_schema,
            )
            logger.info("Payload index on '%s' ready.", field_name)
        except Exception as exc:
            logger.debug("Payload index '%s' skipped (may already exist): %s", field_name, exc)

    def _ensure_text_index(self, field_name: str) -> None:
        """Create a full-text tokenized index for keyword search."""
        try:
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name=field_name,
                field_schema=TextIndexParams(
                    type="text",
                    tokenizer=TokenizerType.WORD,
                    min_token_len=2,
                    max_token_len=20,
                    lowercase=True,
                ),
            )
            logger.info("Full-text index on '%s' ready.", field_name)
        except Exception as exc:
            logger.debug("Full-text index '%s' skipped (may already exist): %s", field_name, exc)

    # ------------------------------------------------------------------
    # Upsert
    # ------------------------------------------------------------------

    def upsert_vectors(self, vectors: list[dict]) -> None:
        """
        Batch upsert vectors into the collection.

        Each item must have:
            - id (str | None): auto-generated if missing
            - vector (list[float]): embedding
            - payload (dict): metadata
        """
        points = [
            PointStruct(
                id=item.get("id") or str(uuid.uuid4()),
                vector={"dense_vector": item["vector"]},
                payload=item["payload"],
            )
            for item in vectors
        ]
        self.client.upsert(
            collection_name=self.collection_name,
            points=points,
            wait=True,
        )
        logger.info("Upserted %d vectors into '%s'.", len(points), self.collection_name)

    # ------------------------------------------------------------------
    # Dense search
    # ------------------------------------------------------------------

    def search(self, query_vector: list[float], top_k: int) -> list[dict]:
        """
        Semantic similarity search using the dense vector.

        Returns chunks with keys: text_chunk, document_name, page_number,
        chunk_index, score.
        """
        response = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            using="dense_vector",
            limit=top_k,
            with_payload=True,
        )
        return [self._hit_to_dict(hit) for hit in response.points]

    # ------------------------------------------------------------------
    # Full-text keyword search
    # ------------------------------------------------------------------

    def keyword_search(self, query_text: str, top_k: int) -> list[dict]:
        """
        Full-text keyword search using Qdrant's text index on text_chunk.

        Returns chunks that contain the query terms.  Results are unscored
        (score set to 0.5 as a neutral placeholder); final ranking is done
        via RRF in hybrid_search().
        """
        try:
            # Extract meaningful keywords (skip very short words)
            keywords = " ".join(
                w for w in query_text.split() if len(w) >= 3
            )
            if not keywords:
                return []

            records, _ = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=Filter(
                    must=[
                        FieldCondition(
                            key="text_chunk",
                            match=MatchText(text=keywords),
                        )
                    ]
                ),
                limit=top_k,
                with_payload=True,
                with_vectors=False,
            )

            chunks = []
            for record in records:
                payload = record.payload or {}
                chunks.append({
                    "text_chunk": payload.get("text_chunk", ""),
                    "document_name": payload.get("document_name", "unknown"),
                    "page_number": payload.get("page_number", 0),
                    "chunk_index": payload.get("chunk_index", 0),
                    "score": 0.5,  # Neutral placeholder; RRF handles ranking
                    "retrieval_type": "keyword",
                })

            # Sort deterministically so the same query always produces the same
            # keyword-result ranking.  Qdrant scroll() does not guarantee stable
            # ordering, so we impose our own: document name asc, then chunk
            # position asc (earlier in document = higher rank for ties).
            chunks.sort(key=lambda c: (c.get("document_name", ""), c.get("chunk_index", 0)))

            logger.debug(
                "Keyword search for '%s' → %d results.", keywords[:60], len(chunks)
            )
            return chunks

        except Exception as exc:
            logger.warning("Keyword search failed (continuing with dense only): %s", exc)
            return []

    # ------------------------------------------------------------------
    # Hybrid search with Reciprocal Rank Fusion
    # ------------------------------------------------------------------

    def hybrid_search(
        self,
        query_vector: list[float],
        query_text: str,
        top_k: int,
    ) -> list[dict]:
        """
        Hybrid retrieval using dense vector search + full-text keyword search,
        fused with Reciprocal Rank Fusion (RRF).

        Query flow
        ──────────
        Step A — Dense search (semantic similarity):
            self.client.query_points(query=query_vector, using="dense_vector", limit=candidate_k)
            → Returns chunks ranked by cosine similarity to the embedded query.
            → Finds semantically related content even when exact words differ.

        Step B — Keyword search (full-text match):
            self.client.scroll(filter=MatchText(text=query_text), limit=candidate_k)
            → Uses Qdrant's tokenized text index on the text_chunk field.
            → Finds chunks containing the actual query words (e.g. "75%", "BS476").
            → Returns unscored results; their rank comes from Qdrant's index order.

        Step C — RRF fusion:
            For every chunk, compute: rrf_score = Σ 1 / (_RRF_K + rank_i)
            summed over each result list where the chunk appeared.
            → Chunks appearing in BOTH lists get a higher combined score.
            → Keyword-only hits and semantic-only hits both survive in the merged list.
            → Final list is sorted by rrf_score descending and truncated to top_k.

        RRF formula: score(d) = Σ 1 / (K + rank_i(d)),  K = 60 (standard constant)
        """
        # Use a wider candidate window so RRF has sufficient material to fuse
        candidate_k = max(top_k * 2, 20)

        # ── Step A: dense (semantic) search ──────────────────────────────────
        dense_chunks = self.search(query_vector, top_k=candidate_k)
        for c in dense_chunks:
            c["retrieval_type"] = "dense"

        # ── Step B: full-text keyword search ─────────────────────────────────
        keyword_chunks = self.keyword_search(query_text, top_k=candidate_k)

        # ── Step C: RRF fusion ────────────────────────────────────────────────
        merged = self._rrf_merge(dense_chunks, keyword_chunks)

        logger.info(
            "Hybrid search | dense=%d  keyword=%d  merged=%d  returning top %d",
            len(dense_chunks),
            len(keyword_chunks),
            len(merged),
            min(top_k, len(merged)),
        )
        return merged[:top_k]

    @staticmethod
    def _rrf_merge(
        dense_results: list[dict],
        keyword_results: list[dict],
        k: int = _RRF_K,
    ) -> list[dict]:
        """
        Merge two ranked result lists using Reciprocal Rank Fusion.

        Each chunk is identified by (document_name, chunk_index).
        Chunks that appear in both lists get a higher combined score.
        """
        rrf_scores: dict[tuple, float] = {}
        chunk_map: dict[tuple, dict] = {}

        def _add_list(results: list[dict], list_weight: float = 1.0) -> None:
            for rank, chunk in enumerate(results, start=1):
                key = (chunk.get("document_name"), chunk.get("chunk_index"))
                rrf_scores[key] = rrf_scores.get(key, 0.0) + list_weight / (k + rank)
                # Prefer the dense chunk version (has real cosine score)
                if key not in chunk_map or chunk.get("retrieval_type") == "dense":
                    chunk_map[key] = chunk

        _add_list(dense_results, list_weight=1.0)
        _add_list(keyword_results, list_weight=1.0)

        sorted_keys = sorted(rrf_scores, key=lambda x: rrf_scores[x], reverse=True)

        merged = []
        for key in sorted_keys:
            chunk = chunk_map[key].copy()
            chunk["rrf_score"] = round(rrf_scores[key], 6)
            merged.append(chunk)

        return merged

    # ------------------------------------------------------------------
    # Document management
    # ------------------------------------------------------------------

    def get_documents(self) -> list[str]:
        """Return a sorted list of unique document names."""
        try:
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
        """Return True if at least one vector for the document exists."""
        records, _ = self.client.scroll(
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
        return len(records) > 0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _hit_to_dict(hit) -> dict:
        payload = hit.payload or {}
        return {
            "text_chunk": payload.get("text_chunk", ""),
            "document_name": payload.get("document_name", "unknown"),
            "page_number": payload.get("page_number", 0),
            "chunk_index": payload.get("chunk_index", 0),
            "score": round(float(hit.score), 4),
        }

"""Qdrant-backed vector storage for document chunks.

Qdrant is an external service so multiple API workers share one index. Vectors
and payloads are written in bounded batches; the API process never loads the
whole corpus into memory.
"""
import uuid
from collections import OrderedDict

from langchain_core.documents import Document
from qdrant_client import QdrantClient, models

from app.config import settings
from app.embeddings import get_embeddings


class VectorStore:
    def __init__(self) -> None:
        self._embeddings = get_embeddings()
        self._client = QdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key,
            timeout=settings.qdrant_timeout,
        )
        self._collection = settings.qdrant_collection
        dim = len(self._embeddings.embed_query("dimension probe"))
        if not self._client.collection_exists(self._collection):
            self._client.create_collection(
                collection_name=self._collection,
                vectors_config=models.VectorParams(
                    size=dim, distance=models.Distance.COSINE, on_disk=True
                ),
                hnsw_config=models.HnswConfigDiff(m=16, ef_construct=128, on_disk=True),
                optimizers_config=models.OptimizersConfigDiff(indexing_threshold=20_000),
                on_disk_payload=True,
            )
        self._client.create_payload_index(
            self._collection, "doc_id", field_schema=models.PayloadSchemaType.KEYWORD
        )

    def add_document(self, filename: str, chunks: list[Document]) -> str:
        """Embed and upsert chunks in batches, returning the source document ID."""
        doc_id = str(uuid.uuid4())
        batch_size = settings.embedding_batch_size
        try:
            for start in range(0, len(chunks), batch_size):
                batch = chunks[start : start + batch_size]
                vectors = self._embeddings.embed_documents([c.page_content for c in batch])
                points = []
                for offset, (chunk, vector) in enumerate(zip(batch, vectors)):
                    chunk_index = start + offset
                    points.append(
                        models.PointStruct(
                            id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"{doc_id}:{chunk_index}")),
                            vector=vector,
                            payload={
                                "doc_id": doc_id,
                                "filename": filename,
                                "chunk_index": chunk_index,
                                "num_chunks": len(chunks),
                                "text": chunk.page_content,
                            },
                        )
                    )
                self._client.upsert(self._collection, points=points, wait=True)
        except Exception:
            self._client.delete(
                self._collection,
                points_selector=models.FilterSelector(
                    filter=models.Filter(
                        must=[models.FieldCondition(key="doc_id", match=models.MatchValue(value=doc_id))]
                    )
                ),
                wait=True,
            )
            raise
        return doc_id

    def delete_document(self, doc_id: str) -> bool:
        found = self._client.count(
            self._collection,
            count_filter=models.Filter(must=[models.FieldCondition(key="doc_id", match=models.MatchValue(value=doc_id))]),
            exact=True,
        ).count
        if not found:
            return False
        self._client.delete(
            self._collection,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[models.FieldCondition(key="doc_id", match=models.MatchValue(value=doc_id))]
                )
            ),
            wait=True,
        )
        return True

    def search(self, query: str, top_k: int) -> list[dict]:
        vector = self._embeddings.embed_query(query)
        hits = self._client.query_points(
            collection_name=self._collection,
            query=vector,
            limit=top_k,
            with_payload=True,
            search_params=models.SearchParams(hnsw_ef=settings.qdrant_hnsw_ef),
        ).points
        return [
            {
                "doc_id": hit.payload["doc_id"],
                "filename": hit.payload["filename"],
                "chunk_index": hit.payload["chunk_index"],
                "text": hit.payload["text"],
                "score": float(hit.score),
            }
            for hit in hits
        ]

    def list_documents(self) -> list[dict]:
        """Return a bounded document listing; never scan an unbounded corpus."""
        cap = settings.document_list_limit
        docs: OrderedDict[str, dict] = OrderedDict()
        offset = None
        while len(docs) < cap:
            points, offset = self._client.scroll(
                self._collection,
                limit=min(1000, cap * 4),
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            if not points:
                break
            for point in points:
                payload = point.payload or {}
                doc_id = payload.get("doc_id")
                if not doc_id:
                    continue
                if doc_id not in docs:
                    docs[doc_id] = {
                        "doc_id": doc_id,
                        "filename": payload.get("filename", ""),
                        "num_chunks": int(payload.get("num_chunks", 0)),
                    }
                    if len(docs) >= cap:
                        break
                if len(docs) >= cap and offset is None:
                    break
            if offset is None:
                break
        return list(docs.values())


_store: VectorStore | None = None


def get_vector_store() -> VectorStore:
    global _store
    if _store is None:
        _store = VectorStore()
    return _store

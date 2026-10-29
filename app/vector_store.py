"""A persistence layer around LangChain's FAISS vectorstore (inner-product
metric, since embeddings are L2-normalized, so it behaves as cosine
similarity), adding doc_id-scoped delete/list on top of it."""
import threading
import uuid

import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_core.documents import Document

from app.config import settings
from app.embeddings import get_embeddings

_INDEX_NAME = "faiss"


class VectorStore:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._embeddings = get_embeddings()
        self._index_dir = settings.index_path
        self._store = self._load_or_create()

    # ---------- persistence ----------

    def _load_or_create(self) -> FAISS:
        index_file = self._index_dir / f"{_INDEX_NAME}.faiss"
        dim = len(self._embeddings.embed_query("dimension probe"))

        if index_file.exists():
            # allow_dangerous_deserialization is safe here: the index/docstore
            # pickle is written only by this app, to its own local index dir.
            store = FAISS.load_local(
                str(self._index_dir),
                self._embeddings,
                index_name=_INDEX_NAME,
                allow_dangerous_deserialization=True,
            )
          
            return store

        return FAISS(
            embedding_function=self._embeddings,
            index=faiss.IndexFlatIP(dim),
            docstore=InMemoryDocstore(),
            index_to_docstore_id={},
            distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT,
        )


    def _save(self) -> None:
        self._store.save_local(str(self._index_dir), index_name=_INDEX_NAME)

    # ---------- mutation ----------

    def add_document(self, filename: str, chunks: list[Document]) -> str:
        """Adds all chunks of one document to the index. Returns the new doc_id."""
        doc_id = str(uuid.uuid4())
        for i, chunk in enumerate(chunks):
            chunk.metadata.update({"doc_id": doc_id, "filename": filename, "chunk_index": i})

        with self._lock:
            ids = [f"{doc_id}:{i}" for i in range(len(chunks))]
            self._store.add_documents(chunks, ids=ids)
            self._save()
        return doc_id

    def delete_document(self, doc_id: str) -> bool:
        with self._lock:
            ids = [
                doc_id_key
                for doc_id_key, doc in self._store.docstore._dict.items()
                if doc.metadata.get("doc_id") == doc_id
            ]
            if not ids:
                return False
            self._store.delete(ids)
            self._save()
        return True

    # ---------- query ----------

    def search(self, query: str, top_k: int) -> list[dict]:
        with self._lock:
            results = self._store.similarity_search_with_score(query, k=top_k)

        return [
            {
                "doc_id": doc.metadata["doc_id"],
                "filename": doc.metadata["filename"],
                "chunk_index": doc.metadata["chunk_index"],
                "text": doc.page_content,
                "score": float(score),
            }
            for doc, score in results
        ]

    def list_documents(self) -> list[dict]:
        docs: dict[str, dict] = {}
        for doc in self._store.docstore._dict.values():
            doc_id = doc.metadata["doc_id"]
            if doc_id not in docs:
                docs[doc_id] = {"doc_id": doc_id, "filename": doc.metadata["filename"], "num_chunks": 0}
            docs[doc_id]["num_chunks"] += 1
        return list(docs.values())


_store: VectorStore | None = None
_store_lock = threading.Lock()


def get_vector_store() -> VectorStore:
    global _store
    if _store is None:
        with _store_lock:
            if _store is None:
                _store = VectorStore()
    return _store

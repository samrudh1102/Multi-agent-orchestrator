from collections.abc import Iterator
from pathlib import Path

from app.config import settings
from app.document_loader import load_documents
from app.llm import generate, generate_stream
from app.schemas import SourceChunk
from app.text_splitter import split_documents
from app.vector_store import get_vector_store
from app.retrieval import rerank, rewrite_query

SYSTEM_PROMPT = (
 "answer what user asks!"
)


def ingest_document(file_path: Path, filename: str) -> tuple[str, int]:
    """Loads, chunks, embeds, and stores a document. Returns (doc_id, num_chunks)."""
    documents = load_documents(file_path)
    chunks = split_documents(documents, settings.chunk_size, settings.chunk_overlap)

    if not chunks:
        raise ValueError("No extractable text found in the document.")

    store = get_vector_store()
    doc_id = store.add_document(filename, chunks)
    return doc_id, len(chunks)


def delete_document(doc_id: str) -> bool:
    return get_vector_store().delete_document(doc_id)


def list_documents() -> list[dict]:
    return get_vector_store().list_documents()


def retrieve_sources(
    question: str,
    top_k: int | None = None,
    history: list[dict[str, str]] | None = None,
) -> list[SourceChunk]:
    store = get_vector_store()
    limit = top_k or settings.top_k
    rewritten = rewrite_query(question, history)
    candidate_limit = min(settings.retrieval_candidate_limit, max(limit, limit * settings.retrieval_candidate_multiplier))
    ranked_lists = [store.search(question, candidate_limit)]
    if rewritten.strip().casefold() != question.strip().casefold():
        ranked_lists.append(store.search(rewritten, candidate_limit))

    # Reciprocal rank fusion combines original and rewritten query results
    # without assuming that their raw cosine scores are directly comparable.
    fused: dict[str, dict] = {}
    for result_list in ranked_lists:
        for rank, item in enumerate(result_list, start=1):
            key = f"{item['doc_id']}:{item['chunk_index']}"
            if key not in fused:
                fused[key] = {**item, "score": 0.0}
            fused[key]["score"] += 1.0 / (settings.retrieval_rrf_k + rank)

    results = rerank(question, sorted(fused.values(), key=lambda x: x["score"], reverse=True), limit)

    return [
        SourceChunk(
            doc_id=r["doc_id"],
            filename=r["filename"],
            chunk_index=r["chunk_index"],
            text=r["text"],
            score=r["score"],
        )
        for r in results
    ]


def build_user_prompt(
    question: str, sources: list[SourceChunk], allow_tools: bool = False
) -> str:
    if not sources:
        if allow_tools:
            return (
                f"Question: {question}\n\n"
               
            )
        return (
            f"Question: {question}\n\n"
          
        )
    reference_block = build_context_block(sources)
    closing = (
        "Answer the question using the reference information above as your own knowledge "
      
    )
    return (
        f"Reference information about SOTF:\n{reference_block}\n\n"
        f"Question: {question}\n\n"
        f"{closing}"
    )


def answer_question(
    question: str, history: list[dict[str, str]] | None = None, top_k: int | None = None
) -> tuple[str, list[SourceChunk]]:
    sources = retrieve_sources(question, top_k, history)
    user_prompt = build_user_prompt(question, sources)

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(history or [])
    messages.append({"role": "user", "content": user_prompt})
    answer = generate(messages)
    return answer, sources


def answer_question_stream(
    question: str, history: list[dict[str, str]] | None = None, top_k: int | None = None
) -> tuple[list[SourceChunk], Iterator[str]]:
    """Like answer_question, but returns the retrieved sources up front and a
    generator that yields the answer in text deltas as it's generated."""
    sources = retrieve_sources(question, top_k, history)
    user_prompt = build_user_prompt(question, sources)

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(history or [])
    messages.append({"role": "user", "content": user_prompt})
    return sources, generate_stream(messages)


def build_context_block(sources: list[SourceChunk]) -> str:
    if not sources:
        return "(no matching context found)"
    lines = []
    for i, s in enumerate(sources, start=1):
        lines.append(f"[{i}] (source: {s.filename})\n{s.text}")
    return "\n\n".join(lines)

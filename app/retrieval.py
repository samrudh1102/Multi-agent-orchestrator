"""Query rewriting, rank fusion and cross-encoder reranking."""
import logging
from functools import lru_cache

from app.config import settings
from app.llm import generate

logger = logging.getLogger(__name__)


def rewrite_query(question: str, history: list[dict[str, str]] | None = None) -> str:
    """Turn a follow-up into a standalone search query; fall back safely."""
    if not settings.query_rewriter_enabled:
        return question
    prior = [m for m in (history or [])[-6:] if m.get("role") in {"user", "assistant"}]
    history_text = "\n".join(f'{m["role"]}: {m.get("content", "")[:1500]}' for m in prior)
    prompt = (
        "Rewrite the latest user question as one concise, standalone search query. "
        "Resolve references using conversation context. Preserve names, identifiers, "
        "dates, and constraints. Do not answer the question. Output only the query.\n\n"
        f"Conversation:\n{history_text or '(no prior conversation)'}\n\nLatest question: {question}"
    )
    try:
        rewritten = generate([
            {"role": "system", "content": "You rewrite questions for document retrieval."},
            {"role": "user", "content": prompt},
        ]).strip().strip('"')
        if rewritten and len(rewritten) <= 1000:
            return rewritten
    except Exception:
        logger.exception("Query rewrite failed; using the original question")
    return question


@lru_cache(maxsize=1)
def _cross_encoder():
    from sentence_transformers import CrossEncoder

    return CrossEncoder(settings.reranker_model, device=settings.reranker_device)


def rerank(query: str, candidates: list[dict], limit: int) -> list[dict]:
    """Rerank a bounded candidate set; preserve vector order on model failure."""
    if not candidates:
        return []
    if not settings.reranker_enabled:
        return candidates[:limit]
    try:
        pairs = [(query, item["text"]) for item in candidates]
        scores = _cross_encoder().predict(pairs, batch_size=settings.reranker_batch_size)
        ranked = []
        for item, score in zip(candidates, scores):
            result = dict(item)
            result["score"] = float(score)
            ranked.append(result)
        return sorted(ranked, key=lambda x: x["score"], reverse=True)[:limit]
    except Exception:
        logger.exception("Cross-encoder reranking failed; using fused vector ranking")
        return candidates[:limit]

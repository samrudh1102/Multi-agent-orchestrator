"""Wraps Google Gemini embedding model for vector generation."""

from functools import lru_cache

from langchain_google_genai import GoogleGenerativeAIEmbeddings

from app.config import settings


@lru_cache(maxsize=1)
def get_embeddings() -> GoogleGenerativeAIEmbeddings:
    return GoogleGenerativeAIEmbeddings(
        model=settings.embedding_model_name,
        google_api_key=settings.google_api_key,
    )
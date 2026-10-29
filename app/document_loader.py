"""Load uploaded files into LangChain Documents via LangChain's document loaders."""
from pathlib import Path

from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader, TextLoader
from langchain_core.documents import Document

_LOADER_CLASSES = {
    ".pdf": PyPDFLoader,
    ".docx": Docx2txtLoader,
    ".txt": TextLoader,
    ".md": TextLoader,
}


class UnsupportedFileTypeError(ValueError):
    pass


def load_documents(file_path: Path) -> list[Document]:
    """Extract LangChain Documents out of a pdf, docx, txt, or md file."""
    suffix = file_path.suffix.lower()
    loader_cls = _LOADER_CLASSES.get(suffix)
    if loader_cls is None:
        raise UnsupportedFileTypeError(
            f"Unsupported file type '{suffix}'. Supported types: {sorted(_LOADER_CLASSES)}"
        )

    if loader_cls is TextLoader:
        return loader_cls(str(file_path), autodetect_encoding=True).load()
    return loader_cls(str(file_path)).load()

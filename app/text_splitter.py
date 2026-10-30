from langchain_core.documents import Document
def split_documents(documents: list[Document], chunk_size: int, chunk_overlap: int) -> list[Document]:
    """Split on semantic text boundaries using token-counted chunk sizes.

    Separators preserve paragraphs, headings, sentences and words where
    possible. Token sizing keeps chunks aligned with model context budgets.
    """
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="cl100k_base",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ", ""],
        add_start_index=True,
    )
    return splitter.split_documents(documents)

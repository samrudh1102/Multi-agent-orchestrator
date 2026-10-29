from pydantic import BaseModel, Field


class UploadResponse(BaseModel):
    doc_id: str
    filename: str
    num_chunks: int


class DocumentInfo(BaseModel):
    doc_id: str
    filename: str
    num_chunks: int


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1)
    top_k: int | None = None
    history: list[dict[str, str]] = Field(
        default_factory=list,
        description='Optional prior turns: [{"role": "user"|"assistant", "content": "..."}]',
    )


class SourceChunk(BaseModel):
    doc_id: str
    filename: str
    chunk_index: int
    text: str
    score: float


class ToolCallInfo(BaseModel):
    name: str
    arguments: dict
    result: str


class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceChunk]
    tool_calls: list[ToolCallInfo] = []

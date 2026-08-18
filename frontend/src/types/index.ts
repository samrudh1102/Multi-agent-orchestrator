export interface SourceChunk {
  doc_id: string;
  filename: string;
  chunk_index: number;
  text: string;
  score: number;
}

export interface ToolCallInfo {
  name: string;
  arguments: Record<string, any>;
  result: string;
}

export interface ToolStatusEvent {
  stage: 'tool_calling' | 'tool_complete';
  name: string;
  display: string;
}

export interface DocumentInfo {
  doc_id: string;
  filename: string;
  num_chunks: number;
}

export interface ChatMessage {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  sources?: SourceChunk[];
  toolCalls?: ToolCallInfo[];
  activeStatus?: ToolStatusEvent | null;
  isStreaming?: boolean;
  error?: string;
}
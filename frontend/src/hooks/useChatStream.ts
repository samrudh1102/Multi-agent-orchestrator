import { useState, useRef, useCallback } from 'react';
import { ChatMessage, SourceChunk, ToolCallInfo, ToolStatusEvent } from '../types';

const STREAM_URL = 'http://localhost:8000/chat/stream';

export function useChatStream() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [isGenerating, setIsGenerating] = useState(false);
  const abortControllerRef = useRef<AbortController | null>(null);

  const stopGenerating = useCallback(() => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
      abortControllerRef.current = null;
      setIsGenerating(false);

      setMessages((prev) =>
        prev.map((msg) =>
          msg.isStreaming ? { ...msg, isStreaming: false, activeStatus: null } : msg
        )
      );
    }
  }, []);

  const clearChat = useCallback(() => {
    stopGenerating();
    setMessages([]);
  }, [stopGenerating]);

  const sendMessage = useCallback(
    async (question: string) => {
      if (!question.trim() || isGenerating) return;

      const userMsgId = crypto.randomUUID();
      const assistantMsgId = crypto.randomUUID();

      const userMessage: ChatMessage = {
        id: userMsgId,
        role: 'user',
        content: question.trim(),
      };

      const assistantMessage: ChatMessage = {
        id: assistantMsgId,
        role: 'assistant',
        content: '',
        sources: [],
        toolCalls: [],
        activeStatus: null,
        isStreaming: true,
      };

      // Formulate past multi-turn history (clean role + content pairs)
      const history = messages
        .filter((m) => !m.error)
        .map((m) => ({
          role: m.role,
          content: m.content,
        }));

      setMessages((prev) => [...prev, userMessage, assistantMessage]);
      setIsGenerating(true);

      const controller = new AbortController();
      abortControllerRef.current = controller;

      try {
        const response = await fetch(STREAM_URL, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            question: question.trim(),
            history,
          }),
          signal: controller.signal,
        });

        if (!response.ok) {
          throw new Error(`HTTP error ${response.status}: ${response.statusText}`);
        }

        if (!response.body) {
          throw new Error('Response body is null');
        }

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;

          buffer += decoder.decode(value, { stream: true });
          const frames = buffer.split('\n\n');
          buffer = frames.pop() || ''; // Keep partial frame in buffer

          for (const frame of frames) {
            if (!frame.trim()) continue;

            const lines = frame.split('\n');
            let event = '';
            let dataStr = '';

            for (const line of lines) {
              if (line.startsWith('event: ')) {
                event = line.substring(7).trim();
              } else if (line.startsWith('data: ')) {
                dataStr = line.substring(6).trim();
              }
            }

            if (!event || !dataStr) continue;

            try {
              const data = JSON.parse(dataStr);

              setMessages((prev) =>
                prev.map((msg) => {
                  if (msg.id !== assistantMsgId) return msg;

                  switch (event) {
                    case 'status':
                      return {
                        ...msg,
                        activeStatus: data as ToolStatusEvent,
                      };
                    case 'sources':
                      return {
                        ...msg,
                        sources: data as SourceChunk[],
                      };
                    case 'delta':
                      return {
                        ...msg,
                        content: msg.content + (data.text || ''),
                      };
                    case 'tool_calls':
                      return {
                        ...msg,
                        toolCalls: data as ToolCallInfo[],
                        activeStatus: null,
                      };
                    case 'error':
                      return {
                        ...msg,
                        error: data.detail || 'An unknown error occurred.',
                        isStreaming: false,
                        activeStatus: null,
                      };
                    case 'done':
                      return {
                        ...msg,
                        isStreaming: false,
                        activeStatus: null,
                      };
                    default:
                      return msg;
                  }
                })
              );
            } catch (err) {
              console.error('Failed to parse SSE JSON:', err, dataStr);
            }
          }
        }
      } catch (err: any) {
        if (err.name === 'AbortError') return;

        setMessages((prev) =>
          prev.map((msg) =>
            msg.id === assistantMsgId
              ? {
                  ...msg,
                  error: err.message || 'Connection failed',
                  isStreaming: false,
                  activeStatus: null,
                }
              : msg
          )
        );
      } finally {
        setIsGenerating(false);
        abortControllerRef.current = null;
        setMessages((prev) =>
          prev.map((msg) => (msg.id === assistantMsgId ? { ...msg, isStreaming: false } : msg))
        );
      }
    },
    [messages, isGenerating]
  );

  return {
    messages,
    isGenerating,
    sendMessage,
    stopGenerating,
    clearChat,
  };
}
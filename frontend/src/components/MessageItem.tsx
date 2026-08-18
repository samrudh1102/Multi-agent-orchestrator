import React from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import rehypeHighlight from 'rehype-highlight';
import { User, Bot, AlertCircle } from 'lucide-react';
import { ChatMessage } from '../types';
import { ToolBadge } from './ToolBadge';
import { ToolAccordion } from './ToolAccordion';
import { SourceCitations } from './SourceCitations';

interface MessageItemProps {
  message: ChatMessage;
  onRetry?: () => void;
}

export const MessageItem: React.FC<MessageItemProps> = ({ message, onRetry }) => {
  const isUser = message.role === 'user';

  return (
    <div className={`flex gap-3.5 ${isUser ? 'justify-end' : 'justify-start'} group`}>
      {!isUser && (
        <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-indigo-500 to-cyan-500 flex items-center justify-center shrink-0 shadow-md shadow-indigo-500/10">
          <Bot className="w-4 h-4 text-white" />
        </div>
      )}

      <div className={`max-w-[85%] sm:max-w-[75%] space-y-2 ${isUser ? 'order-first' : ''}`}>
        {/* User Message Bubble */}
        {isUser ? (
          <div className="px-4 py-2.5 rounded-2xl rounded-tr-none bg-indigo-600 text-white text-sm shadow-md leading-relaxed select-text">
            {message.content}
          </div>
        ) : (
          <div className="space-y-2">
            {/* Live Streaming Tool Badge */}
            {message.activeStatus && (
              <div className="mb-2">
                <ToolBadge status={message.activeStatus} />
              </div>
            )}

            {/* Error Banner */}
            {message.error && (
              <div className="p-3 rounded-lg bg-red-950/40 border border-red-500/30 flex items-start gap-2.5 text-xs text-red-200">
                <AlertCircle className="w-4 h-4 text-red-400 shrink-0 mt-0.5" />
                <div className="space-y-1">
                  <p className="font-semibold">Query Failed</p>
                  <p className="text-red-300/80">{message.error}</p>
                </div>
              </div>
            )}

            {/* Markdown Content Bubble */}
            {(message.content || message.isStreaming) && (
              <div className="p-4 rounded-2xl rounded-tl-none bg-surface border border-border-subtle text-slate-200 text-sm shadow-md prose-custom select-text">
                <ReactMarkdown
                  remarkPlugins={[remarkGfm]}
                  rehypePlugins={[rehypeHighlight]}
                >
                  {message.content}
                </ReactMarkdown>

                {/* Subtle typing cursor during streaming */}
                {message.isStreaming && !message.content && !message.activeStatus && (
                  <div className="flex items-center gap-1.5 py-1">
                    <span className="w-2 h-2 rounded-full bg-indigo-400 animate-bounce [animation-delay:-0.3s]"></span>
                    <span className="w-2 h-2 rounded-full bg-cyan-400 animate-bounce [animation-delay:-0.15s]"></span>
                    <span className="w-2 h-2 rounded-full bg-indigo-400 animate-bounce"></span>
                  </div>
                )}
              </div>
            )}

            {/* Completed Tool Calls Accordion */}
            {message.toolCalls && message.toolCalls.length > 0 && (
              <ToolAccordion toolCalls={message.toolCalls} />
            )}

            {/* Sources / Citations */}
            {message.sources && message.sources.length > 0 && (
              <SourceCitations sources={message.sources} />
            )}
          </div>
        )}
      </div>

      {isUser && (
        <div className="w-8 h-8 rounded-lg bg-slate-800 border border-slate-700 flex items-center justify-center shrink-0">
          <User className="w-4 h-4 text-slate-300" />
        </div>
      )}
    </div>
  );
};
import React, { useState, useRef, useEffect } from 'react';
import {
  Send,
  Square,
  Trash2,
  Menu,
  Sparkles,
  Bot,
  Layers,
} from 'lucide-react';
import { ChatMessage } from '../types';
import { MessageItem } from './MessageItem';
import { QuickChips } from './QuickChips';

interface ChatContainerProps {
  messages: ChatMessage[];
  isGenerating: boolean;
  onSendMessage: (msg: string) => void;
  onStop: () => void;
  onClear: () => void;
  onToggleSidebar: () => void;
}

export const ChatContainer: React.FC<ChatContainerProps> = ({
  messages,
  isGenerating,
  onSendMessage,
  onStop,
  onClear,
  onToggleSidebar,
}) => {
  const [input, setInput] = useState('');
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || isGenerating) return;
    onSendMessage(input);
    setInput('');
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e);
    }
  };

  const handleInput = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setInput(e.target.value);
    e.target.style.height = 'auto';
    e.target.style.height = `${Math.min(e.target.scrollHeight, 160)}px`;
  };

  return (
    <div className="flex-1 flex flex-col h-full bg-[#0a0d14] relative overflow-hidden">
      {/* Top Navbar */}
      <header className="h-14 border-b border-border-subtle px-4 flex items-center justify-between bg-[#0d121f]/80 backdrop-blur-md shrink-0">
        <div className="flex items-center gap-3">
          <button
            onClick={onToggleSidebar}
            className="p-1.5 rounded-lg text-slate-400 hover:text-white hover:bg-slate-800 transition-colors"
          >
            <Menu className="w-5 h-5" />
          </button>
          <div className="flex items-center gap-2">
            <div className="w-7 h-7 rounded-lg bg-indigo-600/20 border border-indigo-500/30 flex items-center justify-center text-indigo-400">
              <Bot className="w-4 h-4" />
            </div>
            <div>
              <h1 className="text-sm font-semibold text-slate-100 flex items-center gap-2">
                SOTF Assistant
                <span className="text-[10px] font-mono px-1.5 py-0.2 rounded bg-indigo-500/20 text-indigo-300 border border-indigo-500/30">
                  LangGraph MCP
                </span>
              </h1>
            </div>
          </div>
        </div>

        <div className="flex items-center gap-2">
          {messages.length > 0 && (
            <button
              onClick={onClear}
              title="Clear Conversation"
              className="flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg text-xs font-medium text-slate-400 hover:text-red-400 hover:bg-red-950/30 border border-transparent hover:border-red-500/20 transition-all"
            >
              <Trash2 className="w-3.5 h-3.5" />
              <span className="hidden sm:inline">Clear</span>
            </button>
          )}
        </div>
      </header>

      {/* Main Chat Scroll Area */}
      <div className="flex-1 overflow-y-auto px-4 py-6">
        <div className="max-w-3xl mx-auto h-full flex flex-col justify-between">
          {messages.length === 0 ? (
            <QuickChips onSelectPrompt={onSendMessage} />
          ) : (
            <div className="space-y-6 pb-4">
              {messages.map((msg) => (
                <MessageItem key={msg.id} message={msg} />
              ))}
              <div ref={messagesEndRef} />
            </div>
          )}
        </div>
      </div>

      {/* Bottom Input Form */}
      <div className="p-4 bg-[#0d121f]/90 border-t border-border-subtle backdrop-blur-md shrink-0">
        <div className="max-w-3xl mx-auto">
          <form onSubmit={handleSubmit} className="relative flex items-end gap-2">
            <div className="relative flex-1 bg-surface rounded-xl border border-border-subtle focus-within:border-indigo-500/50 shadow-inner">
              <textarea
                ref={textareaRef}
                value={input}
                onChange={handleInput}
                onKeyDown={handleKeyDown}
                rows={1}
                placeholder="Ask about test failures, PUDO metrics, ML predictions, or Airflow DAGs..."
                className="w-full bg-transparent px-4 py-3 text-sm text-slate-100 placeholder:text-slate-500 resize-none outline-none max-h-40 min-h-[44px]"
              />
            </div>

            {isGenerating ? (
              <button
                type="button"
                onClick={onStop}
                className="h-11 px-4 rounded-xl bg-red-600/90 hover:bg-red-600 text-white font-medium text-xs flex items-center gap-1.5 shadow-md shadow-red-600/20 transition-all shrink-0"
              >
                <Square className="w-3.5 h-3.5 fill-current" />
                <span>Stop</span>
              </button>
            ) : (
              <button
                type="submit"
                disabled={!input.trim()}
                className="h-11 px-4 rounded-xl bg-indigo-600 hover:bg-indigo-500 disabled:opacity-40 disabled:hover:bg-indigo-600 text-white font-medium text-xs flex items-center justify-center shadow-md shadow-indigo-600/20 transition-all shrink-0"
              >
                <Send className="w-4 h-4" />
              </button>
            )}
          </form>

          <p className="text-center text-[10px] text-slate-500 mt-2 font-mono">
            Directly connected to ClickHouse, Databricks, Jira, Airflow, TestRail & Postgres MCP servers.
          </p>
        </div>
      </div>
    </div>
  );
};
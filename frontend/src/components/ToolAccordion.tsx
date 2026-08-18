import React, { useState } from 'react';
import { ChevronDown, ChevronRight, Terminal, Check, Copy } from 'lucide-react';
import { ToolCallInfo } from '../types';

interface ToolAccordionProps {
  toolCalls: ToolCallInfo[];
}

export const ToolAccordion: React.FC<ToolAccordionProps> = ({ toolCalls }) => {
  const [isOpen, setIsOpen] = useState(false);
  const [copiedIndex, setCopiedIndex] = useState<number | null>(null);

  if (!toolCalls || toolCalls.length === 0) return null;

  const handleCopy = (text: string, index: number) => {
    navigator.clipboard.writeText(text);
    setCopiedIndex(index);
    setTimeout(() => setCopiedIndex(null), 2000);
  };

  return (
    <div className="mt-3 border border-slate-800 rounded-lg overflow-hidden bg-[#0c101b]">
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="w-full flex items-center justify-between px-3.5 py-2 text-xs font-mono text-slate-400 hover:text-slate-200 hover:bg-slate-800/40 transition-colors"
      >
        <div className="flex items-center gap-2">
          <Terminal className="w-3.5 h-3.5 text-cyan-400" />
          <span>
            {toolCalls.length} Tool Execution{toolCalls.length > 1 ? 's' : ''} Performed
          </span>
        </div>
        {isOpen ? <ChevronDown className="w-4 h-4" /> : <ChevronRight className="w-4 h-4" />}
      </button>

      {isOpen && (
        <div className="p-3 space-y-3 border-t border-slate-800 bg-[#090d16]">
          {toolCalls.map((tool, idx) => (
            <div key={idx} className="rounded border border-slate-800/80 bg-slate-950 p-2.5 space-y-2">
              <div className="flex items-center justify-between text-xs">
                <span className="font-mono font-semibold text-indigo-400">{tool.name}</span>
                <button
                  onClick={() => handleCopy(JSON.stringify(tool, null, 2), idx)}
                  className="flex items-center gap-1 text-[10px] text-slate-500 hover:text-slate-300"
                >
                  {copiedIndex === idx ? (
                    <>
                      <Check className="w-3 h-3 text-emerald-400" />
                      <span className="text-emerald-400">Copied</span>
                    </>
                  ) : (
                    <>
                      <Copy className="w-3 h-3" />
                      <span>JSON</span>
                    </>
                  )}
                </button>
              </div>

              {/* Arguments */}
              <div>
                <span className="text-[10px] font-mono uppercase tracking-wider text-slate-500 block mb-1">
                  Arguments
                </span>
                <pre className="p-2 rounded bg-slate-900/90 text-[11px] font-mono text-cyan-300 overflow-x-auto border border-slate-800/50">
                  {JSON.stringify(tool.arguments, null, 2)}
                </pre>
              </div>

              {/* Results */}
              <div>
                <span className="text-[10px] font-mono uppercase tracking-wider text-slate-500 block mb-1">
                  Response Output
                </span>
                <pre className="p-2 rounded bg-slate-900/90 text-[11px] font-mono text-slate-300 overflow-x-auto max-h-48 border border-slate-800/50">
                  {tool.result}
                </pre>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};
import React, { useState } from 'react';
import { BookOpen, ExternalLink, ChevronDown, ChevronRight } from 'lucide-react';
import { SourceChunk } from '../types';

interface SourceCitationsProps {
  sources: SourceChunk[];
}

export const SourceCitations: React.FC<SourceCitationsProps> = ({ sources }) => {
  const [isExpanded, setIsExpanded] = useState(false);

  if (!sources || sources.length === 0) return null;

  return (
    <div className="mt-3 border border-slate-800/80 rounded-lg overflow-hidden bg-slate-900/30">
      <button
        onClick={() => setIsExpanded(!isExpanded)}
        className="w-full flex items-center justify-between px-3 py-1.5 text-xs text-slate-400 hover:text-slate-200 transition-colors"
      >
        <div className="flex items-center gap-1.5 font-mono">
          <BookOpen className="w-3.5 h-3.5 text-indigo-400" />
          <span>Knowledge Base Sources ({sources.length})</span>
        </div>
        {isExpanded ? <ChevronDown className="w-3.5 h-3.5" /> : <ChevronRight className="w-3.5 h-3.5" />}
      </button>

      {isExpanded && (
        <div className="p-2.5 space-y-2 border-t border-slate-800/80 bg-slate-950/60">
          {sources.map((s, idx) => (
            <div key={idx} className="p-2 rounded bg-slate-900/80 border border-slate-800 text-xs space-y-1">
              <div className="flex items-center justify-between text-indigo-300 font-mono text-[11px]">
                <span className="truncate max-w-[240px]">
                  [{idx + 1}] {s.filename}
                </span>
                <span className="text-slate-500">Chunk {s.chunk_index}</span>
              </div>
              <p className="text-slate-400 text-[11px] leading-relaxed line-clamp-3 hover:line-clamp-none transition-all cursor-pointer">
                {s.text}
              </p>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};
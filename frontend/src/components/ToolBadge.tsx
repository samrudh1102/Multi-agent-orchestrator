import React from 'react';
import { Loader2, CheckCircle2, Wrench } from 'lucide-react';
import { ToolStatusEvent } from '../types';

interface ToolBadgeProps {
  status: ToolStatusEvent;
}

export const ToolBadge: React.FC<ToolBadgeProps> = ({ status }) => {
  const isComplete = status.stage === 'tool_complete';

  return (
    <div className="inline-flex items-center gap-2 px-3 py-1.5 rounded-full bg-slate-900/90 border border-indigo-500/30 text-xs font-mono text-indigo-300 shadow-sm backdrop-blur-md animate-in fade-in slide-in-from-top-1 duration-200">
      {isComplete ? (
        <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400" />
      ) : (
        <Loader2 className="w-3.5 h-3.5 text-cyan-400 animate-spin" />
      )}
      <span>{status.display || status.name}</span>
    </div>
  );
};
import React from 'react';
import { Sparkles, Database, Activity, AlertTriangle, GitPullRequest } from 'lucide-react';

interface QuickChipsProps {
  onSelectPrompt: (prompt: string) => void;
}

const CHIPS = [
  {
    icon: Database,
    label: 'Failed ML Predictions',
    prompt: 'Query Databricks for rides with misclassified failure mode predictions and summarize the top error trends.',
  },
  {
    icon: Activity,
    label: 'PUDO Zone Error Rates',
    prompt: 'Show error rate spikes across PUDO zones from ClickHouse logs in the last test runs.',
  },
  {
    icon: GitPullRequest,
    label: 'Airflow Pipeline Status',
    prompt: 'List recent DAG runs for sotf_testrail_sync and check if any sync tasks failed.',
  },
  {
    icon: AlertTriangle,
    label: 'Open Jira Defects',
    prompt: 'Search Jira for active high-priority bugs linked to SOTF scenario execution failures.',
  },
];

export const QuickChips: React.FC<QuickChipsProps> = ({ onSelectPrompt }) => {
  return (
    <div className="max-w-xl mx-auto space-y-4 text-center my-auto py-8 animate-in fade-in duration-500">
      <div className="inline-flex items-center justify-center w-12 h-12 rounded-2xl bg-indigo-600/10 border border-indigo-500/20 text-indigo-400 mb-2">
        <Sparkles className="w-6 h-6" />
      </div>
      <h2 className="text-xl font-semibold text-slate-100">
        Service Orchestration Test Framework
      </h2>
      <p className="text-xs text-slate-400 max-w-md mx-auto">
        Query live logs, trace scenario root causes, inspect Lakehouse ML predictions, and review Airflow pipelines.
      </p>

      <div className="grid grid-cols-1 sm:grid-cols-2 gap-2.5 pt-4 text-left">
        {CHIPS.map((chip, idx) => {
          const Icon = chip.icon;
          return (
            <button
              key={idx}
              onClick={() => onSelectPrompt(chip.prompt)}
              className="p-3 rounded-xl bg-surface border border-border-subtle hover:border-indigo-500/40 hover:bg-surface-subtle transition-all duration-200 group text-left flex flex-col gap-1.5"
            >
              <div className="flex items-center gap-2 text-indigo-400 group-hover:text-indigo-300">
                <Icon className="w-4 h-4" />
                <span className="text-xs font-medium text-slate-200">{chip.label}</span>
              </div>
              <p className="text-[11px] text-slate-400 line-clamp-2 leading-relaxed">
                {chip.prompt}
              </p>
            </button>
          );
        })}
      </div>
    </div>
  );
};
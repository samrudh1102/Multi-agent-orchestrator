import React, { useState, useEffect, useRef } from 'react';
import {
  FileText,
  UploadCloud,
  Trash2,
  X,
  Plus,
  Loader2,
  FolderOpen,
  CheckCircle2,
  AlertCircle,
} from 'lucide-react';
import { DocumentInfo } from '../types';
import { fetchDocuments, uploadDocument, deleteDocument } from '../lib/api';

interface DocumentSidebarProps {
  isOpen: boolean;
  onClose: () => void;
}

export const DocumentSidebar: React.FC<DocumentSidebarProps> = ({ isOpen, onClose }) => {
  const [documents, setDocuments] = useState<DocumentInfo[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [isDragOver, setIsDragOver] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const loadDocs = async () => {
    setIsLoading(true);
    setError(null);
    try {
      const data = await fetchDocuments();
      setDocuments(data);
    } catch (err: any) {
      setError(err.message || 'Failed to load documents');
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    if (isOpen) {
      loadDocs();
    }
  }, [isOpen]);

  const handleUpload = async (file: File) => {
    const allowed = ['.pdf', '.docx', '.txt', '.md'];
    const ext = '.' + file.name.split('.').pop()?.toLowerCase();
    if (!allowed.includes(ext)) {
      setError(`Unsupported file type: ${ext}. Use PDF, DOCX, TXT, or MD.`);
      return;
    }

    setIsUploading(true);
    setError(null);
    try {
      await uploadDocument(file);
      await loadDocs();
    } catch (err: any) {
      setError(err.message || 'Upload failed');
    } finally {
      setIsUploading(false);
    }
  };

  const handleDelete = async (docId: string) => {
    try {
      await deleteDocument(docId);
      setDocuments((prev) => prev.filter((d) => d.doc_id !== docId));
    } catch (err: any) {
      setError(err.message || 'Failed to delete');
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleUpload(e.dataTransfer.files[0]);
    }
  };

  return (
    <>
      {/* Backdrop */}
      {isOpen && (
        <div
          onClick={onClose}
          className="fixed inset-0 bg-black/60 backdrop-blur-sm z-40 lg:hidden"
        />
      )}

      {/* Sidebar Panel */}
      <aside
        className={`fixed top-0 bottom-0 left-0 w-80 bg-[#0d121f] border-r border-border-subtle z-50 flex flex-col transition-transform duration-300 ease-in-out ${
          isOpen ? 'translate-x-0' : '-translate-x-full'
        } lg:relative lg:translate-x-0`}
      >
        {/* Header */}
        <div className="p-4 border-b border-border-subtle flex items-center justify-between">
          <div className="flex items-center gap-2">
            <FolderOpen className="w-5 h-5 text-indigo-400" />
            <h3 className="font-semibold text-sm text-slate-100">Knowledge Base</h3>
          </div>
          <button
            onClick={onClose}
            className="p-1 rounded-md text-slate-400 hover:text-white hover:bg-slate-800 lg:hidden"
          >
            <X className="w-4 h-4" />
          </button>
        </div>

        {/* Drag & Drop Upload Zone */}
        <div className="p-3">
          <input
            type="file"
            ref={fileInputRef}
            onChange={(e) => e.target.files?.[0] && handleUpload(e.target.files[0])}
            accept=".pdf,.docx,.txt,.md"
            className="hidden"
          />
          <div
            onDragOver={(e) => {
              e.preventDefault();
              setIsDragOver(true);
            }}
            onDragLeave={() => setIsDragOver(false)}
            onDrop={handleDrop}
            onClick={() => fileInputRef.current?.click()}
            className={`border-2 border-dashed rounded-xl p-4 text-center cursor-pointer transition-all ${
              isDragOver
                ? 'border-indigo-400 bg-indigo-500/10'
                : 'border-slate-800 hover:border-slate-700 bg-slate-950/40'
            }`}
          >
            {isUploading ? (
              <div className="flex flex-col items-center gap-2 text-indigo-400">
                <Loader2 className="w-6 h-6 animate-spin" />
                <span className="text-xs font-mono">Embedding & indexing...</span>
              </div>
            ) : (
              <div className="flex flex-col items-center gap-1.5 text-slate-400">
                <UploadCloud className="w-6 h-6 text-indigo-400" />
                <span className="text-xs font-medium text-slate-200">Upload Knowledge Doc</span>
                <span className="text-[10px] text-slate-500">PDF, DOCX, TXT, MD</span>
              </div>
            )}
          </div>
        </div>

        {/* Error notification */}
        {error && (
          <div className="mx-3 p-2.5 rounded-lg bg-red-950/40 border border-red-500/30 text-red-200 text-xs flex items-center gap-2">
            <AlertCircle className="w-4 h-4 text-red-400 shrink-0" />
            <span className="truncate">{error}</span>
          </div>
        )}

        {/* Document List */}
        <div className="flex-1 overflow-y-auto p-3 space-y-1.5">
          <div className="text-[11px] font-mono text-slate-500 uppercase tracking-wider px-1 pb-1">
            Indexed Files ({documents.length})
          </div>

          {isLoading ? (
            <div className="flex justify-center py-8 text-slate-500">
              <Loader2 className="w-5 h-5 animate-spin" />
            </div>
          ) : documents.length === 0 ? (
            <div className="text-center py-8 text-xs text-slate-500">
              No files uploaded yet.
            </div>
          ) : (
            documents.map((doc) => (
              <div
                key={doc.doc_id}
                className="group flex items-center justify-between p-2.5 rounded-lg bg-surface border border-border-subtle hover:border-slate-700 transition-all text-xs"
              >
                <div className="flex items-center gap-2.5 min-w-0">
                  <FileText className="w-4 h-4 text-cyan-400 shrink-0" />
                  <div className="min-w-0">
                    <p className="font-medium text-slate-200 truncate">{doc.filename}</p>
                    <p className="text-[10px] font-mono text-slate-500">{doc.num_chunks} chunk(s)</p>
                  </div>
                </div>
                <button
                  onClick={() => handleDelete(doc.doc_id)}
                  title="Delete Document"
                  className="opacity-0 group-hover:opacity-100 p-1 text-slate-500 hover:text-red-400 transition-all"
                >
                  <Trash2 className="w-3.5 h-3.5" />
                </button>
              </div>
            ))
          )}
        </div>

        {/* Footer info */}
        <div className="p-3 border-t border-border-subtle text-[11px] text-slate-500 flex items-center justify-between font-mono">
          <span>Qdrant Vector Database</span>
          <span className="flex items-center gap-1 text-emerald-400">
            <CheckCircle2 className="w-3 h-3" /> Ready
          </span>
        </div>
      </aside>
    </>
  );
};

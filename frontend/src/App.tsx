import React, { useState } from 'react';
import { useChatStream } from './hooks/useChatStream';
import { ChatContainer } from './components/ChatContainer';
import { DocumentSidebar } from './components/DocumentSidebar';

export const App: React.FC = () => {
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const { messages, isGenerating, sendMessage, stopGenerating, clearChat } = useChatStream();

  return (
    <div className="flex h-screen w-screen overflow-hidden bg-[#0a0d14] text-slate-100">
      <DocumentSidebar
        isOpen={sidebarOpen}
        onClose={() => setSidebarOpen(false)}
      />
      <ChatContainer
        messages={messages}
        isGenerating={isGenerating}
        onSendMessage={sendMessage}
        onStop={stopGenerating}
        onClear={clearChat}
        onToggleSidebar={() => setSidebarOpen((prev) => !prev)}
      />
    </div>
  );
};

export default App;
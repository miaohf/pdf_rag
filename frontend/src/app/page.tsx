'use client'

import { useState } from 'react'
import ChatInterface from '@/components/chat/ChatInterface'
import DocumentViewer from '@/components/document/DocumentViewer'
import { apiClient, type DocumentPreviewResponse } from '@/lib/api'

export default function Home() {
  const [currentDocument, setCurrentDocument] = useState<DocumentPreviewResponse | null>(null)
  const [currentChunkId, setCurrentChunkId] = useState<string | null>(null)
  const [isDocumentViewerOpen, setIsDocumentViewerOpen] = useState(false)

  const handleSourceClick = async (filename: string, chunkId: string) => {
    try {
      setCurrentChunkId(chunkId)
      
      const documentData = await apiClient.previewDocument(filename, chunkId, true)
      
      setCurrentDocument(documentData)
      setIsDocumentViewerOpen(true)
    } catch (error) {
      console.error('Error loading document:', error)
      alert(`加载文档失败: ${error instanceof Error ? error.message : '未知错误'}`)
    }
  }

  const handleCloseDocumentViewer = () => {
    setIsDocumentViewerOpen(false)
    setCurrentDocument(null)
    setCurrentChunkId(null)
  }

  return (
    <div className="h-screen bg-background">
      <div className="h-full flex">
        {/* 左侧聊天界面 */}
        <div className={`${isDocumentViewerOpen ? 'w-1/2' : 'w-full'} transition-all duration-300 ease-in-out flex items-start justify-center p-4`}>
          <ChatInterface onSourceClick={handleSourceClick} />
        </div>
        
        {/* 右侧文档查看器 */}
        {isDocumentViewerOpen && (
          <div className="w-1/2 border-l border-border">
            <DocumentViewer
              currentDocument={currentDocument}
              currentChunkId={currentChunkId}
              onClose={handleCloseDocumentViewer}
            />
          </div>
        )}
      </div>
    </div>
  )
} 
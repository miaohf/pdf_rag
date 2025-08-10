'use client'

import { useState } from 'react'
import ChatInterface from '@/components/chat/ChatInterface'
import DocumentViewer from '@/components/document/DocumentViewer'
import { apiClient, type DocumentPreviewResponse } from '@/lib/api'
import { ThemeToggle } from '@/components/ui/theme-toggle'
import { Brain, ChevronsRight, ChevronsLeft } from 'lucide-react'

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
      setIsDocumentViewerCollapsed(false)
    } catch (error) {
      console.error('Error loading document:', error)
      alert(`加载文档失败: ${error instanceof Error ? error.message : '未知错误'}`)
    }
  }

  const [isDocumentViewerCollapsed, setIsDocumentViewerCollapsed] = useState(false)

  const handleCloseDocumentViewer = () => {
    setIsDocumentViewerOpen(false)
    setCurrentDocument(null)
    setCurrentChunkId(null)
    setIsDocumentViewerCollapsed(false)
  }

  const handleCollapseDocumentViewer = () => {
    setIsDocumentViewerCollapsed(true)
  }

  const handleExpandDocumentViewer = () => {
    setIsDocumentViewerCollapsed(false)
  }

  return (
    <div className="h-screen bg-background flex flex-col">
      {/* 全局头部 - 占满页面宽度 */}
      <div className="w-full header-enhanced p-4">
        <div className="flex items-center justify-between px-4">
          <div className="flex items-center gap-3">
            <div className="flex items-center justify-center w-10 h-10 bg-primary rounded-lg">
              <Brain className="w-6 h-6 text-primary-foreground" />
            </div>
            <div>
              <h1 className="text-xl font-semibold text-foreground">法规智能问答</h1>
              <p className="text-sm text-muted-foreground">基于AI的文档问答系统</p>
            </div>
          </div>
          <ThemeToggle />
        </div>
      </div>
      
      <div className="flex-1 flex overflow-hidden">
        {/* 左侧边栏始终固定在最左侧 */}
        <div className="w-64 sidebar-enhanced flex flex-col">
          <div className="p-4 border-b border-border/50">
            <h3 className="text-sm font-medium text-foreground">会话记录</h3>
          </div>
          <div className="flex-1 p-3 space-y-2">
            <button className="w-full p-3 text-left rounded-lg bg-muted/50 hover:bg-muted transition-colors">
              <div className="text-sm font-medium text-foreground truncate">当前会话</div>
              <div className="text-xs text-muted-foreground">刚刚</div>
            </button>
            <button className="w-full p-3 text-left rounded-lg hover:bg-muted/50 transition-colors">
              <div className="text-sm text-foreground truncate">关于法规解释的问题</div>
              <div className="text-xs text-muted-foreground">2小时前</div>
            </button>
            <button className="w-full p-3 text-left rounded-lg hover:bg-muted/50 transition-colors">
              <div className="text-sm text-foreground truncate">合同条款咨询</div>
              <div className="text-xs text-muted-foreground">昨天</div>
            </button>
            <button className="w-full p-3 text-left rounded-lg hover:bg-muted/50 transition-colors">
              <div className="text-sm text-foreground truncate">行政法规查询</div>
              <div className="text-xs text-muted-foreground">3天前</div>
            </button>
          </div>
          <div className="p-3 border-t border-border/50">
            <button className="w-full p-3 btn-primary-enhanced text-primary-foreground rounded-lg text-sm font-medium">
              + 新建会话
            </button>
          </div>
        </div>

        {/* 中间聊天区域：始终渲染，避免状态丢失 */}
        <div className="flex-1 transition-all duration-300 ease-in-out">
          <div className={isDocumentViewerOpen ? 'h-full' : 'h-full max-w-4xl mx-auto p-4'}>
            <ChatInterface onSourceClick={handleSourceClick} />
          </div>
        </div>

        {/* 右侧文档查看器（仅在开启时显示） */}
        {isDocumentViewerOpen && (
          <div className="w-1/3 border-l border-border overflow-hidden relative">
            {!isDocumentViewerCollapsed ? (
              <>
                <button
                  className="absolute top-2 right-2 w-9 h-9 flex items-center justify-center rounded hover:bg-muted transition-colors z-10"
                  onClick={handleCollapseDocumentViewer}
                  title="折叠"
                >
                  <ChevronsRight className="w-4 h-4" />
                </button>
                <DocumentViewer
                  currentDocument={currentDocument}
                  currentChunkId={currentChunkId}
                  onCollapse={handleCollapseDocumentViewer}
                />
              </>
            ) : (
              <>
                <button
                  className="absolute top-2 right-2 w-9 h-9 flex items-center justify-center rounded hover:bg-muted transition-colors z-10"
                  onClick={handleExpandDocumentViewer}
                  title="展开"
                >
                  <ChevronsLeft className="w-4 h-4" />
                </button>
                <div className="h-full w-full bg-card/50" />
              </>
            )}
          </div>
        )}
        
      </div>
    </div>
  )
} 
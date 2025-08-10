'use client'

import { useState, useEffect, useRef } from 'react'
import { X, Download, FileText, Search, ZoomIn, ZoomOut, Eye, Code } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Input } from '@/components/ui/input'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import rehypeRaw from 'rehype-raw'
import rehypeHighlightSearch from '@/lib/rehypeHighlightSearch'
import { apiClient, type DocumentPreviewResponse } from '@/lib/api'

interface DocumentViewerProps {
  currentDocument: DocumentPreviewResponse | null
  currentChunkId: string | null
  onClose: () => void
}

export default function DocumentViewer({
  currentDocument,
  currentChunkId,
  onClose
}: DocumentViewerProps) {
  const [searchTerm, setSearchTerm] = useState('')
  const [fontSize, setFontSize] = useState(14)
  const [viewMode, setViewMode] = useState<'preview' | 'source'>('preview') // 新增视图模式
  const contentRef = useRef<HTMLDivElement>(null)
  const chunkRef = useRef<HTMLDivElement>(null)

  // 检查是否为Markdown文件
  const isMarkdownFile = currentDocument?.filename 
    ? (currentDocument.filename.toLowerCase().endsWith('.md') || 
       currentDocument.filename.toLowerCase().endsWith('.markdown'))
    : false
  
  // 调试信息 (可以在开发时启用)
  // console.log('DocumentViewer Debug:', {
  //   filename: currentDocument?.filename,
  //   isMarkdownFile,
  //   viewMode,
  //   hasContent: !!currentDocument?.content
  // })

  useEffect(() => {
    // 源码模式：滚动到行
    if (currentDocument?.highlight_info && contentRef.current && viewMode === 'source') {
      const targetLine = currentDocument.highlight_info.start_line
      const lineElement = contentRef.current.querySelector(`[data-line="${targetLine}"]`)
      if (lineElement) {
        lineElement.scrollIntoView({ behavior: 'smooth', block: 'center' })
      }
    }
    // 预览模式：滚动到chunk块
    if (viewMode === 'preview' && chunkRef.current) {
      chunkRef.current.scrollIntoView({ behavior: 'smooth', block: 'center' })
    }
  }, [currentDocument, viewMode])

  if (!currentDocument) {
    return (
      <div className="h-full flex items-center justify-center bg-muted/20">
        <p className="text-muted-foreground">请选择要查看的文档</p>
      </div>
    )
  }

  const highlightText = (text: string, line: number) => {
    if (!searchTerm) return text
    
    const regex = new RegExp(`(${searchTerm})`, 'gi')
    return text.replace(regex, '<mark class="bg-yellow-200 dark:bg-yellow-800">$1</mark>')
  }

  // 高亮Markdown内容的函数
  const highlightMarkdownContent = (content: string, searchTerm: string): string => {
    if (!searchTerm) return content
    
    const regex = new RegExp(`(${searchTerm.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')})`, 'gi')
    const result = content.replace(regex, '<mark>$1</mark>')
    
    // 调试信息
    console.log('Highlight Debug:', {
      searchTerm,
      hasMatches: regex.test(content),
      originalLength: content.length,
      highlightedLength: result.length,
      sampleBefore: content.substring(0, 100),
      sampleAfter: result.substring(0, 100)
    })
    
    return result
  }

  // 创建带搜索高亮的组件
  const createHighlightComponents = (searchTerm: string) => {
    const highlightText = (text: string) => {
      if (!searchTerm) return text
      const regex = new RegExp(`(${searchTerm.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')})`, 'gi')
      const parts = text.split(regex)
      return parts.map((part, index) => 
        regex.test(part) ? (
          <mark key={index} className="bg-yellow-200 dark:bg-yellow-800/70 px-1 py-0.5 rounded font-semibold">
            {part}
          </mark>
        ) : part
      )
    }

    return {
      h1: (props: any) => (
        <h1 {...props} className="text-xl font-bold mb-3 text-foreground border-b-2 border-border pb-1">
          {typeof props.children === 'string' ? highlightText(props.children) : props.children}
        </h1>
      ),
      h2: (props: any) => (
        <h2 {...props} className="text-lg font-semibold mb-2 text-foreground border-b border-border pb-1">
          {typeof props.children === 'string' ? highlightText(props.children) : props.children}
        </h2>
      ),
      h3: (props: any) => (
        <h3 {...props} className="text-base font-medium mb-2 text-foreground">
          {typeof props.children === 'string' ? highlightText(props.children) : props.children}
        </h3>
      ),
      p: (props: any) => (
        <p {...props} className="mb-2 text-sm text-foreground leading-relaxed">
          {typeof props.children === 'string' ? highlightText(props.children) : props.children}
        </p>
      ),
      text: (props: any) => {
        if (typeof props.children === 'string') {
          return <>{highlightText(props.children)}</>
        }
        return props.children
      },
      ul: (props: any) => <ul {...props} className="mb-2 space-y-1 list-disc pl-5" />,
      ol: (props: any) => <ol {...props} className="mb-2 space-y-1 list-decimal pl-5" />,
      li: (props: any) => <li {...props} className="text-sm text-foreground leading-relaxed" />,
      code: (props: any) => <code {...props} className="bg-muted px-1.5 py-0.5 rounded text-xs font-mono text-primary" />,
      blockquote: (props: any) => <blockquote {...props} className="border-l-4 border-primary pl-4 italic text-muted-foreground bg-muted/20 py-2 rounded-r" />,
      table: (props: any) => <table {...props} className="min-w-full border-collapse border border-border mb-4 bg-card" />,
      thead: (props: any) => <thead {...props} className="bg-muted" />,
      tbody: (props: any) => <tbody {...props} className="bg-card" />,
      tr: (props: any) => <tr {...props} className="border-b border-border hover:bg-muted/50" />,
      th: (props: any) => <th {...props} className="border border-border px-4 py-2 text-left font-semibold text-foreground bg-muted" />,
      td: (props: any) => <td {...props} className="border border-border px-4 py-2 text-foreground" />,
    }
  }

  // 标准Markdown组件样式（参考backup版本）
  const getStandardMarkdownComponents = () => ({
    h1: (props: any) => <h1 {...props} className="text-xl font-bold mb-3 text-foreground border-b-2 border-border pb-1" />,
    h2: (props: any) => <h2 {...props} className="text-lg font-semibold mb-2 text-foreground border-b border-border pb-1" />,
    h3: (props: any) => <h3 {...props} className="text-base font-medium mb-2 text-foreground" />,
    p: (props: any) => <p {...props} className="mb-2 text-sm text-foreground leading-relaxed" />,
    ul: (props: any) => <ul {...props} className="mb-2 space-y-1 list-disc pl-5" />,
    ol: (props: any) => <ol {...props} className="mb-2 space-y-1 list-decimal pl-5" />,
    li: (props: any) => <li {...props} className="text-sm text-foreground leading-relaxed" />,
    code: (props: any) => <code {...props} className="bg-muted px-1.5 py-0.5 rounded text-xs font-mono text-primary" />,
    blockquote: (props: any) => <blockquote {...props} className="border-l-4 border-primary pl-4 italic text-muted-foreground bg-muted/20 py-2 rounded-r" />,
    table: (props: any) => <table {...props} className="min-w-full border-collapse border border-border mb-4 bg-card" />,
    thead: (props: any) => <thead {...props} className="bg-muted" />,
    tbody: (props: any) => <tbody {...props} className="bg-card" />,
    tr: (props: any) => <tr {...props} className="border-b border-border hover:bg-muted/50" />,
    th: (props: any) => <th {...props} className="border border-border px-4 py-2 text-left font-semibold text-foreground bg-muted" />,
    td: (props: any) => <td {...props} className="border border-border px-4 py-2 text-foreground" />,
  })



  const isHighlightedLine = (lineNumber: number) => {
    if (!currentDocument?.highlight_info) return false

    const hi: any = (currentDocument as any).highlight_info
    // 优先使用字符区间推导行号
    if (typeof hi.char_start === 'number' && typeof hi.char_end === 'number') {
      const content = currentDocument.content || ''
      // 计算 start_line/end_line（与后端一致：按'\n'计数，1-based）
      const start = Math.max(0, Math.min(content.length, hi.char_start))
      const end = Math.max(start, Math.min(content.length, hi.char_end))
      let sLines = 1
      for (let i = 0; i < start; i++) if (content.charCodeAt(i) === 10) sLines++
      let eLines = sLines
      for (let i = start; i < end; i++) if (content.charCodeAt(i) === 10) eLines++
      return lineNumber >= sLines && lineNumber <= eLines
    }

    // 回退使用后端的行号
    return (
      lineNumber >= currentDocument.highlight_info.start_line &&
      lineNumber <= currentDocument.highlight_info.end_line
    )
  }

  return (
    <div className="h-full flex flex-col bg-background">
      {/* 文档头部 */}
      <CardHeader className="border-b border-border">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3 min-w-0 flex-1">
            <FileText className="w-5 h-5 text-primary flex-shrink-0" />
            <div className="min-w-0 flex-1">
              <CardTitle className="text-base truncate">
                {currentDocument.filename}
              </CardTitle>
              {currentDocument.chunk_info && (
                <p className="text-sm text-muted-foreground">
                  区块 ID: {currentDocument.chunk_info.id}
                </p>
              )}
            </div>
          </div>
          <div className="flex items-center gap-2">
            <Button
              size="sm"
              variant="outline"
              onClick={() => window.open(apiClient.getDownloadUrl(currentDocument.filename), '_blank')}
            >
              <Download className="w-4 h-4 mr-1" />
              下载
            </Button>
            <Button size="sm" variant="ghost" onClick={onClose}>
              <X className="w-4 h-4" />
            </Button>
          </div>
        </div>
        
        {/* 搜索和控制栏 */}
        <div className="flex items-center gap-2 mt-4">
          <div className="relative flex-1">
            <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-muted-foreground" />
            <Input
              placeholder="在文档中搜索..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="pl-10"
            />
          </div>
          <div className="flex items-center gap-1">
            {/* Markdown文件的视图模式切换 */}
            {isMarkdownFile && (
              <div className="flex items-center border border-border rounded-md">
                <Button
                  size="sm"
                  variant={viewMode === 'preview' ? 'default' : 'ghost'}
                  onClick={() => setViewMode('preview')}
                  className="rounded-r-none border-r border-border/50"
                >
                  <Eye className="w-4 h-4 mr-1" />
                  预览
                </Button>
                <Button
                  size="sm"
                  variant={viewMode === 'source' ? 'default' : 'ghost'}
                  onClick={() => setViewMode('source')}
                  className="rounded-l-none"
                >
                  <Code className="w-4 h-4 mr-1" />
                  源码
                </Button>
              </div>
            )}

            <Button
              size="sm"
              variant="outline"
              onClick={() => setFontSize(Math.max(12, fontSize - 1))}
            >
              <ZoomOut className="w-4 h-4" />
            </Button>
            <span className="text-sm text-muted-foreground px-2">{fontSize}px</span>
            <Button
              size="sm"
              variant="outline"
              onClick={() => setFontSize(Math.min(20, fontSize + 1))}
            >
              <ZoomIn className="w-4 h-4" />
            </Button>
          </div>
        </div>
      </CardHeader>

      {/* 文档内容 */}
      <CardContent className="flex-1 overflow-y-auto p-0">
        <div
          ref={contentRef}
          className="p-6"
          style={{ fontSize: `${fontSize}px` }}
        >
          {isMarkdownFile && viewMode === 'preview' ? (
            /* Markdown预览模式（带片段高亮）*/
            <div className="prose prose-sm max-w-4xl dark:prose-invert">
              {(() => {
                const content = currentDocument.content || ''
                // 优先使用 highlight_info 的字符区间
                const hi = (currentDocument as any)?.highlight_info as any
                let start = typeof hi?.char_start === 'number' ? hi.char_start : -1
                let end = typeof hi?.char_end === 'number' ? hi.char_end : -1

                // 备选1：后端在chunk_info中返回了字符区间
                if ((start < 0 || end <= start) && (currentDocument as any)?.chunk_info) {
                  const startChar = (currentDocument as any)?.chunk_info?.start_char
                  const endChar = (currentDocument as any)?.chunk_info?.end_char
                  if (typeof startChar === 'number' && typeof endChar === 'number' && endChar > startChar) {
                    start = startChar
                    end = endChar
                  }
                }

                // 备选2：没有字符范围，则通过chunk内容在全文中定位
                if ((start < 0 || end <= start) && currentDocument.chunk_info?.content) {
                  const cleanFull = content
                  const cleanChunk = currentDocument.chunk_info.content.trim()
                  const direct = cleanFull.indexOf(cleanChunk)
                  if (direct >= 0) {
                    start = direct
                    end = direct + cleanChunk.length
                  } else {
                    const anchor = cleanChunk.substring(0, Math.min(100, cleanChunk.length))
                    const anchorIdx = cleanFull.indexOf(anchor)
                    if (anchorIdx >= 0) {
                      start = anchorIdx
                      end = Math.min(anchorIdx + cleanChunk.length * 1.2, cleanFull.length)
                    }
                  }
                }

                if (start >= 0 && end > start && end <= content.length) {
                  const before = content.slice(0, start)
                  const highlight = content.slice(start, end)
                  const after = content.slice(end)

                  return (
                    <div>
                      {before && (
                        <ReactMarkdown remarkPlugins={[remarkGfm]} rehypePlugins={[rehypeRaw, [rehypeHighlightSearch, { searchTerm }]]} components={getStandardMarkdownComponents()}>
                          {before}
                        </ReactMarkdown>
                      )}

                      <div ref={chunkRef} className="bg-yellow-100 text-gray-900 dark:text-gray-900 border-l-4 border-yellow-500 pl-4 py-4 my-6 rounded-r-lg ring-1 ring-yellow-200">
                        <div className="flex items-center gap-2 mb-3">
                          <span className="text-xs bg-yellow-500 text-white px-2 py-1 rounded">🎯 引用片段</span>
                          {currentDocument.chunk_info?.id && (
                            <span className="text-xs border border-yellow-400 text-yellow-700 px-2 py-1 rounded">ID: {currentDocument.chunk_info.id}</span>
                          )}
                        </div>
                        <ReactMarkdown remarkPlugins={[remarkGfm]} rehypePlugins={[rehypeRaw, [rehypeHighlightSearch, { searchTerm }]]} components={getStandardMarkdownComponents()}>
                          {highlight}
                        </ReactMarkdown>
                      </div>

                      {after && (
                        <ReactMarkdown remarkPlugins={[remarkGfm]} rehypePlugins={[rehypeRaw, [rehypeHighlightSearch, { searchTerm }]]} components={getStandardMarkdownComponents()}>
                          {after}
                        </ReactMarkdown>
                      )}
                    </div>
                  )
                }

                // 没有字符位置信息时，显示整文+搜索高亮
                return (
                  <ReactMarkdown remarkPlugins={[remarkGfm]} rehypePlugins={[rehypeRaw, [rehypeHighlightSearch, { searchTerm }]]} components={getStandardMarkdownComponents()}>
                    {content}
                  </ReactMarkdown>
                )
              })()}
              
              {/* 调试：显示处理后的内容 */}
              {searchTerm && (
                <details className="mt-4 p-4 border border-red-500 bg-red-50 dark:bg-red-900/20">
                  <summary className="cursor-pointer text-red-600 dark:text-red-400">调试信息 (点击展开)</summary>
                  <pre className="mt-2 text-xs overflow-auto max-h-40 bg-white dark:bg-gray-800 p-2 rounded">
                    {highlightMarkdownContent(currentDocument.content, searchTerm).substring(0, 500)}...
                  </pre>
                </details>
              )}
            </div>
          ) : currentDocument.content_lines ? (
            /* 源码模式（带行号） */
            <div className="space-y-1 font-mono">
              {currentDocument.content_lines.map((line, index) => {
                const lineNumber = index + 1
                const isHighlighted = isHighlightedLine(lineNumber)
                
                return (
                  <div
                    key={index}
                    data-line={lineNumber}
                    className={`flex ${isHighlighted ? 'bg-yellow-200 dark:bg-yellow-800/30 border-l-4 border-yellow-500' : ''}`}
                  >
                    <span className="inline-block w-12 text-muted-foreground text-right pr-4 select-none">
                      {lineNumber}
                    </span>
                    <span
                      className="flex-1 whitespace-pre-wrap break-words"
                      dangerouslySetInnerHTML={{
                        __html: highlightText(line, lineNumber)
                      }}
                    />
                  </div>
                )
              })}
            </div>
          ) : (
            /* 普通文本模式 */
            <div className="prose dark:prose-invert max-w-none">
              <div
                dangerouslySetInnerHTML={{
                  __html: highlightText(currentDocument.content, 0)
                }}
              />
            </div>
          )}
        </div>
      </CardContent>

      {/* 底部信息 */}
      <div className="border-t border-border p-3 bg-muted/50">
        <div className="flex justify-between items-center text-sm text-muted-foreground">
          <span>
            {currentDocument.content_lines ? 
              `共 ${currentDocument.total_lines} 行` : 
              `文档长度: ${currentDocument.content.length} 字符`
            }
          </span>
          {currentDocument.chunk_info && (
            <span>
              区块范围: {currentDocument.chunk_info.start_line} - {currentDocument.chunk_info.end_line}
            </span>
          )}
        </div>
      </div>
    </div>
  )
} 
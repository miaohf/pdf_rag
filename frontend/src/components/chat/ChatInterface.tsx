'use client'

import { useState, useRef, useEffect } from 'react'
import { Send, FileText, Download, ChevronDown, ChevronRight, ChevronUp, Eye } from 'lucide-react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { Button } from '@/components/ui/button'
// removed Input; using native textarea for auto-resize
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'

import { apiClient, type Source, type QueryResponse } from '@/lib/api'

interface Message {
  id: string
  type: 'user' | 'assistant'
  content: string
  sources?: Source[]
  timestamp?: number
  isLoading?: boolean
  thinking?: string
}

interface ChatInterfaceProps {
  onSourceClick: (filename: string, chunkId: string) => void
}

export default function ChatInterface({ onSourceClick }: ChatInterfaceProps) {
  const [messages, setMessages] = useState<Message[]>([
    {
      id: '1',
      type: 'assistant',
      content: '您好！我是PDF文档智能问答助手。您可以向我提问关于已上传文档的任何问题，我会基于文档内容为您提供准确的答案。\n\n**使用提示：**\n- 提出具体的问题以获得更准确的答案\n- 点击答案中的引用链接可以查看原文档\n- 您可以下载引用的文档进行详细阅读\n- 点击"思考过程"可以查看AI的分析思路',
      // 初始消息不带时间，避免SSR与CSR时间不一致
    }
  ])
  const [inputValue, setInputValue] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [expandedThinking, setExpandedThinking] = useState<Set<string>>(new Set())
  const [isInputHidden, setIsInputHidden] = useState(false)
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const inputRef = useRef<HTMLTextAreaElement>(null)

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }

  useEffect(() => {
    scrollToBottom()
  }, [messages])

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!inputValue.trim() || isLoading) return

    const now = Date.now()

    const userMessage: Message = {
      id: Date.now().toString(),
      type: 'user',
      content: inputValue.trim(),
      timestamp: now,
    }

    const loadingMessage: Message = {
      id: (Date.now() + 1).toString(),
      type: 'assistant',
      content: 'thinking',
      timestamp: now,
      isLoading: true,
    }

    setMessages(prev => [...prev, userMessage, loadingMessage])
    setInputValue('')
    setIsLoading(true)

    try {
      const response: QueryResponse = await apiClient.query({
        question: userMessage.content,
        max_chunks: 5,
      })

      // 提取思考过程
      let thinking = ''
      if (response.answer.includes('<think>') && response.answer.includes('</think>')) {
        const thinkMatch = response.answer.match(/<think>([\s\S]*?)<\/think>/);
        if (thinkMatch) {
          thinking = thinkMatch[1].trim()
          // 从答案中移除思考标签
          response.answer = response.answer.replace(/<think>[\s\S]*?<\/think>/, '').trim()
        }
      }

      const assistantMessage: Message = {
        id: (Date.now() + 2).toString(),
        type: 'assistant',
        content: response.answer,
        sources: response.sources,
        thinking: thinking || undefined,
        timestamp: Date.now(),
      }

      setMessages(prev => prev.slice(0, -1).concat([assistantMessage]))
    } catch (error) {
      const errorMessage: Message = {
        id: (Date.now() + 2).toString(),
        type: 'assistant',
        content: `抱歉，查询时出现错误：${error instanceof Error ? error.message : '未知错误'}`,
        timestamp: Date.now(),
      }
      setMessages(prev => prev.slice(0, -1).concat([errorMessage]))
    } finally {
      setIsLoading(false)
    }
  }

  // 自适应高度
  const autoResizeTextarea = (el: HTMLTextAreaElement | null) => {
    if (!el) return
    el.style.height = 'auto'
    el.style.height = Math.min(el.scrollHeight, 160) + 'px' // 上限约为10行（具体取决于line-height）
  }

  useEffect(() => {
    autoResizeTextarea(inputRef.current)
  }, [inputValue])

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSubmit(e as any)
    }
  }

  const toggleThinking = (messageId: string) => {
    setExpandedThinking(prev => {
      const newSet = new Set(prev)
      if (newSet.has(messageId)) {
        newSet.delete(messageId)
      } else {
        newSet.add(messageId)
      }
      return newSet
    })
  }

  return (
    <div className="h-full flex flex-col bg-background transition-all duration-300">
      {/* 消息列表 */}
      <div className="flex-1 scroll-container p-6 space-y-4 w-full overflow-y-auto">
        {messages.map((message) => (
          <div key={message.id} className={`flex ${message.type === 'user' ? 'justify-end' : 'justify-start'}`}>
            <div className={`max-w-[80%] ${message.type === 'user' ? 'order-2' : 'order-1'}`}>
              <Card className={`chat-message enhanced-card animate-fade-in ${message.type === 'user' 
                ? 'bg-primary/10 text-foreground dark:bg-primary dark:text-primary-foreground border border-primary/20' 
                : 'bg-card dark:bg-transparent'} ${message.isLoading ? 'animate-pulse' : ''}`}>
                <CardContent className="px-4 py-3">
                  <div className="prose prose-sm max-w-none dark:prose-invert prose-p:my-2 prose-p:first:mt-0 prose-p:last:mb-0 prose-headings:my-2 prose-ul:my-2 prose-ol:my-2">
                    <ReactMarkdown remarkPlugins={[remarkGfm]}>
                      {message.content}
                    </ReactMarkdown>
                  </div>
                  
                  {/* 思考过程 */}
                  {message.thinking && message.type === 'assistant' && (
                    <div className="mt-4 pt-4 border-t border-border/30">
                      <button
                        onClick={() => toggleThinking(message.id)}
                        className="flex items-center gap-2 text-sm text-muted-foreground hover:text-foreground transition-colors group"
                        title={expandedThinking.has(message.id) ? "隐藏思考过程" : "查看AI思考过程"}
                      >
                        <Eye className="w-4 h-4" /> 
                        thinking message
                        
                        {/* {expandedThinking.has(message.id) ? (
                          <ChevronDown className="w-3 h-3" />
                        ) : (
                          <ChevronRight className="w-3 h-3" />
                        )} */}
                        {/* <span className="opacity-0 group-hover:opacity-100 transition-opacity duration-200">
                          Show
                        </span> */}
                      </button>
                      
                      {expandedThinking.has(message.id) && (
                        <div className="mt-2 p-3 thinking-area-enhanced rounded-md transition-all duration-200">
                          <div className="prose prose-sm max-w-none dark:prose-invert text-muted-foreground scroll-container max-h-96 prose-p:my-2 prose-p:first:mt-0 prose-p:last:mb-0 prose-headings:my-2">
                            <ReactMarkdown remarkPlugins={[remarkGfm]}>
                              {message.thinking}
                            </ReactMarkdown>
                          </div>
                        </div>
                      )}
                    </div>
                  )}
                  
                  {/* 源文档引用 */}
                  {message.sources && message.sources.length > 0 && (
                    <div className="mt-4 pt-4 border-t border-border/30">
                      <p className="text-sm font-medium text-muted-foreground mb-2">Reference Documents:</p>
                      <div className="space-y-2">
                        {message.sources.map((source, index) => (
                          <Card key={index} className="border border-border/20 bg-muted/50">
                            <CardContent className="p-3">
                              <div className="flex items-start justify-between gap-2">
                                <div className="flex-1 min-w-0">
                                  <div className="flex items-center gap-2 mb-1">
                                    <FileText className="w-4 h-4 text-primary" />
                                    <span className="text-sm font-medium truncate">
                                      {source.filename}
                                    </span>
                                  </div>
                                  <div className="text-xs text-muted-foreground">
                                    <span className="mr-3">分片ID: {source.chunk_id}</span>
                                    <span>相似度: {source.similarity.toFixed(3)}</span>
                                  </div>
                                </div>
                                <div className="flex gap-1">
                                  <Button
                                    size="sm"
                                    variant="ghost"
                                    onClick={() => onSourceClick(source.filename, source.chunk_id)}
                                    className="h-8 px-2 bg-transparent hover:bg-transparent"
                                    title="查看原文档"
                                  >
                                    <Eye className="w-3 h-3" />
                                  </Button>
                                  <Button
                                    size="sm"
                                    variant="ghost"
                                    onClick={() => window.open(apiClient.getDownloadUrl(source.filename), '_blank')}
                                    className="h-8 px-2 bg-transparent hover:bg-transparent"
                                    title="下载文档"
                                  >
                                    <Download className="w-3 h-3" />
                                  </Button>
                                </div>
                              </div>
                            </CardContent>
                          </Card>
                        ))}
                      </div>
                    </div>
                  )}
                </CardContent>
              </Card>
              {/* 时间戳显示（可选） */}
              {message.timestamp && (
                <div className={`text-xs text-muted-foreground mt-1 px-1 text-right`}>
                  {new Date(message.timestamp).toLocaleTimeString()}
                </div>
              )}
            </div>
          </div>
        ))}
        <div ref={messagesEndRef} />
      </div>

      {/* 输入区域 */}
      <div className="w-full border-t border-border/20 bg-card/50 backdrop-blur-sm">
        {/* 切换按钮 */}
        <div className="flex justify-center py-1">
          <Button
            variant="ghost"
            size="sm"
            onClick={() => setIsInputHidden(!isInputHidden)}
            className="text-muted-foreground hover:text-foreground transition-all duration-200 rounded-full h-6 w-6 p-0"
          >
            {isInputHidden ? (
              <ChevronUp className="w-3 h-3" />
            ) : (
              <ChevronDown className="w-3 h-3" />
            )}
          </Button>
        </div>

        {/* 输入框区域 */}
        {!isInputHidden && (
          <div className="px-6 pb-6 pt-2">
            <form onSubmit={handleSubmit} className="flex gap-3 px-4">
              <textarea
                ref={inputRef}
                value={inputValue}
                onChange={(e) => {
                  setInputValue(e.target.value)
                  autoResizeTextarea(e.target)
                }}
                onKeyDown={handleKeyDown}
                placeholder="输入您的问题..."
                disabled={isLoading}
                rows={1}
                className="flex-1 input-enhanced rounded-xl px-4 py-3 text-sm placeholder:text-muted-foreground/60 focus:outline-none max-h-40 overflow-y-auto"
              />
              <Button 
                type="submit" 
                disabled={!inputValue.trim() || isLoading}
                className="btn-primary-enhanced rounded-xl px-4 py-3"
              >
                <Send className="w-4 h-4" />
              </Button>
            </form>
            <p className="text-xs text-muted-foreground/60 mt-3 text-center px-4">
              Enter 发送 • Shift + Enter 换行
            </p>
          </div>
        )}
      </div>
    </div>
  )
} 
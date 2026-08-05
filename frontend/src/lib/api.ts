const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'

export interface QueryRequest {
  question: string
  max_chunks?: number
  max_tokens?: number
  temperature?: number
  top_k?: number
}

export interface Source {
  chunk_id: string
  filename: string
  document_name?: string
  similarity: number
  content_preview: string
  content_length: number
  metadata: Record<string, any>
  reference_format: string
  location_data: {
    file_path: string
    chunk_id: string
    start_line?: number
    end_line?: number
    start_char?: number
    end_char?: number
    anchor_text: string
  }
  preview_url: string
  download_url: string
}

export interface QueryResponse {
  answer: string
  confidence: number
  sources: Source[]
  response_time: number
  query_id: string
}

export interface DocumentPreviewResponse {
  filename: string
  content: string
  content_lines: string[]
  total_lines: number
  full_document_content?: string
  chunk_info?: {
    id: string
    content: string
    original_content?: string
    start_line?: number
    end_line?: number
    parent_chunk_id?: string
    start_char?: number
    end_char?: number
  }
  highlight_info?: {
    start_line: number
    end_line: number
    text: string
    char_start?: number
    char_end?: number
  }
}

export class ApiClient {
  private baseUrl: string

  constructor(baseUrl: string = API_BASE_URL) {
    this.baseUrl = baseUrl
  }

  async query(request: QueryRequest): Promise<QueryResponse> {
    const response = await fetch(`${this.baseUrl}/api/query`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(request),
    })

    if (!response.ok) {
      throw new Error(`API Error: ${response.status} ${response.statusText}`)
    }

    return response.json()
  }

  async previewDocument(
    filename: string,
    chunkId?: string,
    includeHighlight: boolean = false
  ): Promise<DocumentPreviewResponse> {
    const params = new URLSearchParams()
    if (chunkId) params.append('chunk_id', chunkId)
    if (includeHighlight) params.append('include_highlight', 'true')

    const response = await fetch(
      `${this.baseUrl}/api/documents/preview/${encodeURIComponent(filename)}?${params}`,
      {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
        },
      }
    )

    if (!response.ok) {
      throw new Error(`API Error: ${response.status} ${response.statusText}`)
    }

    return response.json()
  }

  getDownloadUrl(filename: string): string {
    return `${this.baseUrl}/api/documents/download/${encodeURIComponent(filename)}`
  }
}

export const apiClient = new ApiClient() 
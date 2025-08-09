#!/usr/bin/env python3
"""
PDF RAG Web API 服务

提供RESTful API接口的Web服务。
"""

import time
import traceback
from typing import List, Optional, Dict, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, UploadFile, File, Depends, BackgroundTasks, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel

from src.utils.config import Config
from src.utils.logger import Logger, get_logger
from src.core.database import Database

# 全局变量
config = None
db = None
rag_service = None
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    global config, db, rag_service
    
    try:
        # 启动时初始化
        logger.info("🚀 启动 PDF RAG Web API 服务")
        
        # 加载配置
        config = Config()
        Logger.setup(config)
        
        # 初始化数据库
        db = Database(config)
        
        # 初始化RAG服务
        from src.rag.service import RAGService
        rag_service = RAGService(config, db)
        
        logger.info("✅ 服务初始化完成")
        yield
        
    except Exception as e:
        logger.error(f"❌ 服务初始化失败: {e}")
        raise
    finally:
        # 关闭时清理
        logger.info("🛑 关闭 PDF RAG Web API 服务")
        if db:
            db.close()


# 创建FastAPI应用
app = FastAPI(
    title="PDF RAG API",
    description="法律法规智能问答系统 API",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境中应该配置具体的域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# 数据模型
class QueryRequest(BaseModel):
    """查询请求模型"""
    question: str
    max_chunks: Optional[int] = None
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_k: Optional[int] = None


class ConversationQueryRequest(BaseModel):
    """多轮对话查询请求模型"""
    question: str
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_k: Optional[int] = None


class QueryResponse(BaseModel):
    """查询响应模型"""
    answer: str
    confidence: Optional[float] = None
    sources: List[Dict[str, Any]] = []
    response_time: float
    query_id: Optional[str] = None


class DocumentImportRequest(BaseModel):
    """文档导入请求模型"""
    file_path: str
    process_immediately: bool = True


class DocumentSearchRequest(BaseModel):
    """文档搜索请求模型"""
    query: str
    limit: int = 10


class HealthResponse(BaseModel):
    """健康检查响应模型"""
    status: str
    version: str
    timestamp: float
    services: Dict[str, bool]


class StatsResponse(BaseModel):
    """统计信息响应模型"""
    documents: int
    chunks: int
    queries: int
    processed_documents: int
    processing_rate: str


# 依赖注入
def get_config() -> Config:
    """获取配置"""
    if config is None:
        raise HTTPException(status_code=500, detail="服务未初始化")
    return config


def get_database() -> Database:
    """获取数据库"""
    if db is None:
        raise HTTPException(status_code=500, detail="数据库未初始化")
    return db


def get_rag_service():
    """获取RAG服务"""
    if rag_service is None:
        raise HTTPException(status_code=500, detail="RAG服务未初始化")
    return rag_service


# API路由
@app.get("/health", response_model=HealthResponse)
async def health_check(
    config: Config = Depends(get_config),
    database: Database = Depends(get_database)
):
    """健康检查"""
    try:
        # 检查数据库
        db_ok = database.check_connection()
        
        # 检查Ollama
        ollama_ok = False
        try:
            import httpx
            response = httpx.get(f"{config.ollama.url}/api/tags", timeout=5)
            ollama_ok = response.status_code == 200
        except Exception:
            pass
        
        services = {
            "database": db_ok,
            "ollama": ollama_ok,
        }
        
        status = "healthy" if all(services.values()) else "unhealthy"
        
        return HealthResponse(
            status=status,
            version="0.1.0",
            timestamp=time.time(),
            services=services
        )
        
    except Exception as e:
        logger.error(f"健康检查失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/query", response_model=QueryResponse)
async def query(
    request: QueryRequest,
    rag: Any = Depends(get_rag_service)
):
    """标准查询接口"""
    try:
        start_time = time.time()
        
        # 执行查询
        result = await rag.query_async(
            question=request.question,
            max_chunks=request.max_chunks,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_k=request.top_k
        )
        
        response_time = time.time() - start_time
        
        return QueryResponse(
            answer=result.get('answer', ''),
            confidence=result.get('confidence'),
            sources=result.get('sources', []),
            response_time=response_time,
            query_id=result.get('query_id')
        )
        
    except Exception as e:
        logger.error(f"查询失败: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"查询失败: {str(e)}")


@app.post("/api/query/stream")
async def query_stream(
    request: QueryRequest,
    rag: Any = Depends(get_rag_service)
):
    """流式查询接口"""
    try:
        async def generate():
            async for chunk in rag.query_stream_async(
                question=request.question,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                top_k=request.top_k
            ):
                yield f"data: {chunk}\n\n"
            yield "data: [DONE]\n\n"
        
        return StreamingResponse(
            generate(),
            media_type="text/stream-server",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            }
        )
        
    except Exception as e:
        logger.error(f"流式查询失败: {e}")
        raise HTTPException(status_code=500, detail=f"流式查询失败: {str(e)}")


@app.post("/api/documents/import")
async def import_document(
    request: DocumentImportRequest,
    background_tasks: BackgroundTasks,
    database: Database = Depends(get_database),
    config: Config = Depends(get_config)
):
    """导入文档"""
    try:
        from src.document.processor import DocumentProcessor
        processor = DocumentProcessor(config, database)
        
        if request.process_immediately:
            # 立即处理
            doc_ids = processor.process_file(request.file_path)
            return {
                "message": "文档导入并处理完成",
                "document_ids": doc_ids,
                "chunk_count": len(doc_ids)
            }
        else:
            # 后台处理
            background_tasks.add_task(processor.process_file, request.file_path)
            return {
                "message": "文档已添加到处理队列",
                "status": "processing"
            }
            
    except Exception as e:
        logger.error(f"文档导入失败: {e}")
        raise HTTPException(status_code=500, detail=f"文档导入失败: {str(e)}")


@app.post("/api/documents/upload")
async def upload_document(
    file: UploadFile = File(...),
    process_immediately: bool = True,
    background_tasks: BackgroundTasks = None,
    database: Database = Depends(get_database),
    config: Config = Depends(get_config)
):
    """上传并导入文档"""
    try:
        # 保存上传的文件
        import tempfile
        import shutil
        from pathlib import Path
        
        # 检查文件类型
        allowed_types = ['.pdf', '.md', '.txt']
        file_extension = Path(file.filename).suffix.lower()
        if file_extension not in allowed_types:
            raise HTTPException(
                status_code=400, 
                detail=f"不支持的文件类型: {file_extension}"
            )
        
        # 创建临时文件
        with tempfile.NamedTemporaryFile(
            delete=False, 
            suffix=file_extension
        ) as temp_file:
            shutil.copyfileobj(file.file, temp_file)
            temp_path = temp_file.name
        
        # 处理文档
        from src.document.processor import DocumentProcessor
        processor = DocumentProcessor(config, database)
        
        if process_immediately:
            doc_ids = processor.process_file(temp_path)
            # 清理临时文件
            Path(temp_path).unlink()
            
            return {
                "message": "文档上传并处理完成",
                "filename": file.filename,
                "document_ids": doc_ids,
                "chunk_count": len(doc_ids)
            }
        else:
            # 后台处理
            background_tasks.add_task(
                lambda: processor.process_file(temp_path) and Path(temp_path).unlink()
            )
            return {
                "message": "文档已上传并添加到处理队列",
                "filename": file.filename,
                "status": "processing"
            }
            
    except Exception as e:
        logger.error(f"文档上传失败: {e}")
        raise HTTPException(status_code=500, detail=f"文档上传失败: {str(e)}")


@app.get("/api/documents/search")
async def search_documents(
    query: str,
    limit: int = 10,
    database: Database = Depends(get_database)
):
    """搜索文档"""
    try:
        # 这里应该实现文档搜索逻辑
        # 暂时返回示例数据
        return {
            "query": query,
            "results": [],
            "total": 0,
            "limit": limit
        }
        
    except Exception as e:
        logger.error(f"文档搜索失败: {e}")
        raise HTTPException(status_code=500, detail=f"文档搜索失败: {str(e)}")


@app.get("/api/documents/{doc_id}")
async def get_document_info(
    doc_id: str,
    database: Database = Depends(get_database)
):
    """获取文档信息"""
    try:
        from src.document.processor import DocumentProcessor
        processor = DocumentProcessor(database=database)
        doc_info = processor.get_document_info(doc_id)
        
        if not doc_info:
            raise HTTPException(status_code=404, detail="文档未找到")
        
        return doc_info
        
    except Exception as e:
        logger.error(f"获取文档信息失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取文档信息失败: {str(e)}")


@app.get("/api/documents/preview/{filename}")
async def preview_document(
    filename: str,
    chunk_id: Optional[str] = None,
    include_highlight: bool = False,
    database: Database = Depends(get_database)
):
    """文档预览API - 支持定位到特定chunk"""
    try:
        from pathlib import Path
        import os
        
        # 安全检查文件路径
        safe_filename = os.path.basename(filename)
        file_path = Path("docs") / safe_filename
        
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="文档文件未找到")
        
        # 读取文档内容
        with open(file_path, 'r', encoding='utf-8') as f:
            full_content = f.read()
        
        # 默认返回完整文档内容
        display_content = full_content
        
        # 如果指定了chunk_id，获取chunk位置信息和高亮位置
        chunk_info = None
        highlight_info = None
        
        if chunk_id:
            with database.get_session() as session:
                from src.core.models import DocumentChunk
                chunk = session.query(DocumentChunk).filter(
                    DocumentChunk.id == chunk_id
                ).first()
                
                if chunk:
                    chunk_info = {
                        'id': chunk.id,
                        'content': chunk.content,
                        'original_content': getattr(chunk, 'original_content', chunk.content),
                        'start_line': getattr(chunk, 'start_line', None),
                        'end_line': getattr(chunk, 'end_line', None),
                        'start_char': getattr(chunk, 'start_char', None),
                        'end_char': getattr(chunk, 'end_char', None)
                    }
                    
                    # 如果需要高亮且有位置信息，计算高亮范围
                    if include_highlight and chunk.content:
                        # 尝试在完整文档中找到chunk内容的位置
                        chunk_text = chunk.content.strip()
                        
                        # 先尝试精确匹配
                        chunk_start_pos = full_content.find(chunk_text)
                        
                        # 如果精确匹配失败，尝试找到chunk中的关键句子
                        if chunk_start_pos == -1:
                            # 分割chunk为行，尝试找到第一行和最后一行
                            chunk_lines = chunk_text.split('\n')
                            if chunk_lines:
                                first_line = chunk_lines[0].strip()
                                if first_line and len(first_line) > 10:  # 确保行足够长
                                    chunk_start_pos = full_content.find(first_line)
                                    if chunk_start_pos != -1:
                                        # 找到起始位置，计算结束位置
                                        last_line = chunk_lines[-1].strip() if len(chunk_lines) > 1 else first_line
                                        last_line_pos = full_content.find(last_line, chunk_start_pos)
                                        if last_line_pos != -1:
                                            chunk_end_pos = last_line_pos + len(last_line)
                                        else:
                                            chunk_end_pos = chunk_start_pos + len(first_line)
                        
                        if chunk_start_pos != -1:
                            # 计算起始和结束行号
                            content_before_chunk = full_content[:chunk_start_pos]
                            start_line = content_before_chunk.count('\n') + 1
                            
                            # 找到chunk结束位置
                            if 'chunk_end_pos' not in locals():
                                chunk_end_pos = chunk_start_pos + len(chunk_text)
                            
                            content_to_end = full_content[:chunk_end_pos]
                            end_line = content_to_end.count('\n') + 1
                            
                            highlight_info = {
                                'start_line': start_line,
                                'end_line': end_line,
                                'text': chunk_text[:100] + '...' if len(chunk_text) > 100 else chunk_text
                            }
                            
                            logger.info(f"计算高亮范围: {start_line}-{end_line}行")
                        else:
                            logger.warning(f"无法在完整文档中找到chunk内容: {chunk_text[:50]}...")
                    
                    logger.info(f"返回完整文档内容，长度: {len(full_content)}")
        
        response_data = {
            'filename': safe_filename,
            'content': full_content,  # 始终返回完整文档内容
            'content_lines': full_content.split('\n'),
            'total_lines': len(full_content.split('\n')),
            'full_document_content': full_content,  # 保持向后兼容
            'chunk_info': chunk_info,
            'highlight_info': highlight_info  # 使用计算出的高亮信息
        }
        
        return response_data
        
    except Exception as e:
        logger.error(f"文档预览失败: {e}")
        raise HTTPException(status_code=500, detail=f"文档预览失败: {str(e)}")


@app.get("/api/documents/{document_id}/chunks")
async def get_document_chunks(
    document_id: str,
    summary_mode: bool = Query(False, description="是否返回摘要模式"),
    summary_length: int = Query(100, description="摘要长度"),
    offset: int = Query(0, description="偏移量"),
    limit: int = Query(50, description="每页数量"),
    database: Database = Depends(get_database)
):
    """
    获取文档片段列表 - 支持摘要模式动态加载
    
    Args:
        document_id: 文档ID
        summary_mode: 是否返回摘要模式（仅返回内容前N个字符）
        summary_length: 摘要长度
        offset: 偏移量
        limit: 每页数量
    """
    try:
        with database.get_session() as session:
            from sqlalchemy import text
            
            # 基础查询
            query = text("""
                SELECT 
                    c.id,
                    c.chunk_index,
                    c.start_char,
                    c.end_char,
                    CASE 
                        WHEN :summary_mode THEN 
                            CASE 
                                WHEN LENGTH(c.content) > :summary_length THEN 
                                    SUBSTRING(c.content, 1, :summary_length) || '...'
                                ELSE c.content
                            END
                        ELSE c.content
                    END as content,
                    LENGTH(c.content) as full_length,
                    c.metadata
                FROM document_chunks c
                WHERE c.document_id = :document_id
                ORDER BY c.chunk_index
                OFFSET :offset LIMIT :limit
            """)
            
            chunks = session.execute(query, {
                "document_id": document_id,
                "summary_mode": summary_mode,
                "summary_length": summary_length,
                "offset": offset,
                "limit": limit
            }).fetchall()
            
            # 获取总数
            count_query = text("""
                SELECT COUNT(*) FROM document_chunks 
                WHERE document_id = :document_id
            """)
            total_count = session.execute(count_query, {"document_id": document_id}).scalar()
            
            result = []
            for chunk in chunks:
                chunk_data = {
                    "id": chunk.id,
                    "chunk_index": chunk.chunk_index,
                    "start_char": chunk.start_char,
                    "end_char": chunk.end_char,
                    "content": chunk.content,
                    "metadata": chunk.metadata or {},
                    "is_summary": summary_mode and chunk.full_length > summary_length,
                    "full_length": chunk.full_length
                }
                result.append(chunk_data)
            
            return {
                "chunks": result,
                "total": total_count,
                "offset": offset,
                "limit": limit,
                "has_more": offset + limit < total_count
            }
            
    except Exception as e:
        logger.error(f"获取文档片段失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取文档片段失败: {str(e)}")


@app.get("/api/chunks/{chunk_id}")
async def get_chunk_detail(
    chunk_id: str,
    database: Database = Depends(get_database)
):
    """
    获取单个片段的完整内容
    
    Args:
        chunk_id: 片段ID
    """
    try:
        with database.get_session() as session:
            from sqlalchemy import text
            
            query = text("""
                SELECT 
                    c.id,
                    c.document_id,
                    c.chunk_index,
                    c.content,
                    c.original_content,
                    c.start_char,
                    c.end_char,
                    c.metadata,
                    d.filename,
                    d.title
                FROM document_chunks c
                JOIN documents d ON c.document_id = d.id
                WHERE c.id = :chunk_id
            """)
            
            chunk = session.execute(query, {"chunk_id": chunk_id}).fetchone()
            
            if not chunk:
                raise HTTPException(status_code=404, detail="片段不存在")
            
            return {
                "id": chunk.id,
                "document_id": chunk.document_id,
                "document_name": chunk.filename,
                "document_title": chunk.title,
                "chunk_index": chunk.chunk_index,
                "content": chunk.content,
                "original_content": chunk.original_content,
                "start_char": chunk.start_char,
                "end_char": chunk.end_char,
                "metadata": chunk.metadata or {}
            }
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取片段详情失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取片段详情失败: {str(e)}")


@app.get("/api/documents/download/{filename}")
async def download_document(filename: str):
    """文档下载API"""
    try:
        from pathlib import Path
        import os
        from fastapi.responses import FileResponse
        
        # 安全检查文件路径
        safe_filename = os.path.basename(filename)
        file_path = Path("docs") / safe_filename
        
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="文档文件未找到")
        
        return FileResponse(
            path=file_path,
            filename=safe_filename,
            media_type='application/octet-stream'
        )
        
    except Exception as e:
        logger.error(f"文档下载失败: {e}")
        raise HTTPException(status_code=500, detail=f"文档下载失败: {str(e)}")


@app.get("/api/models")
async def get_models(config: Config = Depends(get_config)):
    """获取可用模型列表"""
    try:
        import httpx
        response = httpx.get(f"{config.ollama.url}/api/tags", timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            return {
                "models": data.get('models', []),
                "current_model": config.ollama.model
            }
        else:
            return {
                "models": [],
                "current_model": config.ollama.model,
                "error": "无法获取模型列表"
            }
            
    except Exception as e:
        logger.error(f"获取模型列表失败: {e}")
        return {
            "models": [],
            "current_model": config.ollama.model,
            "error": str(e)
        }


@app.post("/api/conversation/query")
async def conversation_query(
    request: ConversationQueryRequest,
    rag: Any = Depends(get_rag_service)
):
    """多轮对话查询接口"""
    try:
        result = rag.query_with_conversation(
            question=request.question,
            session_id=request.session_id,
            user_id=request.user_id,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_k=request.top_k
        )
        
        return result
        
    except Exception as e:
        logger.error(f"多轮对话查询失败: {e}")
        raise HTTPException(status_code=500, detail=f"多轮对话查询失败: {str(e)}")


@app.get("/api/conversation/{session_id}/history")
async def get_conversation_history(
    session_id: str,
    limit: int = 20,
    rag: Any = Depends(get_rag_service)
):
    """获取对话历史"""
    try:
        history = rag.get_conversation_history(session_id, limit)
        return {
            "session_id": session_id,
            "messages": history,
            "total": len(history)
        }
        
    except Exception as e:
        logger.error(f"获取对话历史失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取对话历史失败: {str(e)}")


@app.get("/tools")
async def list_tools(rag: Any = Depends(get_rag_service)):
    """列出可用工具"""
    try:
        tools = rag.list_available_tools()
        return {
            "tools": tools,
            "total": len(tools)
        }
        
    except Exception as e:
        logger.error(f"获取工具列表失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取工具列表失败: {str(e)}")


@app.get("/conversation/stats")
async def get_conversation_stats(rag: Any = Depends(get_rag_service)):
    """获取对话统计信息"""
    try:
        stats = rag.get_conversation_stats()
        return stats
        
    except Exception as e:
        logger.error(f"获取对话统计失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取对话统计失败: {str(e)}")


@app.get("/stats", response_model=StatsResponse)
async def get_stats(database: Database = Depends(get_database)):
    """获取统计信息"""
    try:
        stats = database.get_stats()
        return StatsResponse(**stats)
        
    except Exception as e:
        logger.error(f"获取统计信息失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取统计信息失败: {str(e)}")


# 异常处理
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """全局异常处理"""
    logger.error(f"未处理的异常: {exc}")
    logger.error(traceback.format_exc())
    
    return JSONResponse(
        status_code=500,
        content={
            "detail": "内部服务器错误",
            "type": type(exc).__name__,
            "message": str(exc)
        }
    )


if __name__ == "__main__":
    import uvicorn
    
    # 从配置获取设置
    try:
        config = Config()
        host = getattr(config, 'api_host', '0.0.0.0')
        port = getattr(config, 'api_port', 8000)
    except Exception:
        host = '0.0.0.0'
        port = 8000
    
    logger.info(f"🚀 启动Web服务: http://{host}:{port}")
    logger.info(f"📖 API文档: http://{host}:{port}/docs")
    
    uvicorn.run(
        "app:app",
        host=host,
        port=port,
        reload=True,
        log_level="info"
    ) 
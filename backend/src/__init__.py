"""
PDF RAG - 法律法规智能问答系统

基于RAG技术的智能文档问答系统，专为法律法规文档设计。
"""

__version__ = "0.1.0"
__author__ = "PDF RAG Team"

# 导入主要模块
from src.document.processor import DocumentProcessor
from src.rag.service import RAGService
from src.core.database import Database
from src.utils.config import Config

# 暴露主要类和函数
__all__ = [
    "DocumentProcessor",
    "RAGService", 
    "Database",
    "Config",
    "__version__",
    "__author__"
]

# 创建全局服务实例
_config = None
_rag_service = None

def get_config():
    """获取全局配置实例"""
    global _config
    if _config is None:
        _config = Config()
    return _config

def get_rag_service():
    """获取全局RAG服务实例"""
    global _rag_service
    if _rag_service is None:
        _rag_service = RAGService(get_config())
    return _rag_service

# 便捷的查询函数
def query(question: str, **kwargs):
    """便捷的查询函数"""
    service = get_rag_service()
    return service.query(question, **kwargs)

def query_stream(question: str, **kwargs):
    """便捷的流式查询函数"""
    service = get_rag_service()
    return service.query_stream(question, **kwargs) 
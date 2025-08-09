"""
数据库连接和操作模块

负责PostgreSQL数据库连接、表管理和基本操作。
"""

import asyncio
from typing import List, Dict, Any, Optional, Tuple
from sqlalchemy import create_engine, text, MetaData, Table, Column, Integer, String, Text, DateTime, Float
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import QueuePool
from contextlib import contextmanager, asynccontextmanager
from datetime import datetime
import uuid

from src.utils.config import Config
from src.utils.logger import get_logger
from src.core.models import Base

logger = get_logger(__name__)


class Database:
    """数据库管理器"""
    
    def __init__(self, config: Config):
        """
        初始化数据库连接
        
        Args:
            config: 配置对象
        """
        self.config = config
        self._engine = None
        self._async_engine = None
        self._session_factory = None
        self._async_session_factory = None
        self._metadata = MetaData()
        
        # 初始化连接
        self._init_engines()
        self._define_tables()
        
        logger.info(f"数据库连接初始化完成: {config.database.host}:{config.database.port}")
        
        # 创建ORM表
        self.create_orm_tables()
    
    def _init_engines(self) -> None:
        """初始化数据库引擎"""
        # 同步引擎
        self._engine = create_engine(
            self.config.database.url,
            poolclass=QueuePool,
            pool_size=self.config.database.pool_size,
            max_overflow=self.config.database.max_overflow,
            pool_pre_ping=True,
            echo=False
        )
        
        # 异步引擎
        async_url = self.config.database.url.replace('postgresql://', 'postgresql+asyncpg://')
        self._async_engine = create_async_engine(
            async_url,
            pool_size=self.config.database.pool_size,
            max_overflow=self.config.database.max_overflow,
            pool_pre_ping=True,
            echo=False
        )
        
        # 会话工厂
        self._session_factory = sessionmaker(bind=self._engine)
        self._async_session_factory = async_sessionmaker(
            bind=self._async_engine,
            class_=AsyncSession,
            expire_on_commit=False
        )
    
    def _define_tables(self) -> None:
        """定义数据库表结构"""
        # 文档表
        self.documents_table = Table(
            'documents',
            self._metadata,
            Column('id', String(36), primary_key=True, default=lambda: str(uuid.uuid4())),
            Column('filename', String(255), nullable=False),
            Column('file_path', Text, nullable=False),
            Column('file_size', Integer, nullable=False),
            Column('file_hash', String(64), nullable=False),
            Column('content_type', String(100), nullable=False),
            Column('upload_time', DateTime, default=datetime.utcnow),
            Column('processed', Integer, default=0),  # 0: 未处理, 1: 处理中, 2: 已处理, -1: 处理失败
            Column('chunk_count', Integer, default=0),
            Column('metadata', Text),  # JSON格式的元数据
        )
        
        # 文档片段表
        self.chunks_table = Table(
            'document_chunks',
            self._metadata,
            Column('id', String(36), primary_key=True, default=lambda: str(uuid.uuid4())),
            Column('document_id', String(36), nullable=False),
            Column('chunk_index', Integer, nullable=False),
            Column('content', Text, nullable=False),
            Column('content_length', Integer, nullable=False),
            Column('parent_chunk_id', String(36)),  # 父文档片段ID（用于父子文档模式）
            Column('embedding', Text),  # 向量嵌入（JSON格式）
            Column('metadata', Text),  # 片段元数据（JSON格式）
            Column('created_time', DateTime, default=datetime.utcnow),
        )
        
        # 查询历史表
        self.queries_table = Table(
            'query_history',
            self._metadata,
            Column('id', String(36), primary_key=True, default=lambda: str(uuid.uuid4())),
            Column('question', Text, nullable=False),
            Column('answer', Text, nullable=False),
            Column('sources', Text),  # 参考来源（JSON格式）
            Column('confidence', Float),
            Column('response_time', Float),  # 响应时间（秒）
            Column('query_time', DateTime, default=datetime.utcnow),
            Column('user_id', String(100)),  # 用户ID（可选）
        )
    
    @contextmanager
    def get_session(self):
        """获取同步数据库会话的上下文管理器"""
        session = self._session_factory()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"数据库操作错误: {e}")
            raise
        finally:
            session.close()
    
    @asynccontextmanager
    async def get_async_session(self):
        """获取异步数据库会话的上下文管理器"""
        session = self._async_session_factory()
        try:
            yield session
            await session.commit()
        except Exception as e:
            await session.rollback()
            logger.error(f"异步数据库操作错误: {e}")
            raise
        finally:
            await session.close()
    
    def create_tables(self) -> None:
        """创建数据库表"""
        try:
            # 创建pgvector扩展
            with self._engine.connect() as conn:
                conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
                conn.commit()
            
            # 创建表
            self._metadata.create_all(self._engine)
            logger.info("数据库表创建完成")
            
        except Exception as e:
            logger.error(f"创建数据库表失败: {e}")
            raise
    
    def check_connection(self) -> bool:
        """检查数据库连接"""
        try:
            with self._engine.connect() as conn:
                result = conn.execute(text("SELECT 1"))
                return result.fetchone()[0] == 1
        except Exception as e:
            logger.error(f"数据库连接检查失败: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """获取数据库统计信息"""
        try:
            with self.get_session() as session:
                # 文档数量
                doc_count = session.execute(
                    text("SELECT COUNT(*) FROM documents")
                ).scalar()
                
                # 文档片段数量
                chunk_count = session.execute(
                    text("SELECT COUNT(*) FROM document_chunks")
                ).scalar()
                
                # 查询历史数量
                query_count = session.execute(
                    text("SELECT COUNT(*) FROM query_history")
                ).scalar()
                
                # 已处理文档数量
                processed_docs = session.execute(
                    text("SELECT COUNT(*) FROM documents WHERE processed = 2")
                ).scalar()
                
                return {
                    'documents': doc_count or 0,
                    'chunks': chunk_count or 0,
                    'queries': query_count or 0,
                    'processed_documents': processed_docs or 0,
                    'processing_rate': f"{(processed_docs / max(doc_count, 1)) * 100:.1f}%" if doc_count else "0%"
                }
                
        except Exception as e:
            logger.error(f"获取数据库统计信息失败: {e}")
            return {}
    
    def cleanup_old_data(self, days: int = 30) -> int:
        """
        清理旧数据
        
        Args:
            days: 保留天数
            
        Returns:
            清理的记录数
        """
        try:
            with self.get_session() as session:
                # 清理旧的查询历史
                result = session.execute(
                    text("""
                        DELETE FROM query_history 
                        WHERE query_time < NOW() - INTERVAL :days DAY
                    """),
                    {"days": days}
                )
                
                deleted_count = result.rowcount
                logger.info(f"清理了 {deleted_count} 条旧查询记录")
                return deleted_count
                
        except Exception as e:
            logger.error(f"清理旧数据失败: {e}")
            return 0
    
    def create_orm_tables(self):
        """创建ORM表结构"""
        try:
            # 创建所有表
            Base.metadata.create_all(bind=self._engine)
            logger.info("ORM表结构创建完成")
        except Exception as e:
            logger.error(f"创建ORM表失败: {e}")
            raise
    
    def close(self) -> None:
        """关闭数据库连接"""
        try:
            if self._engine:
                self._engine.dispose()
            if self._async_engine:
                asyncio.run(self._async_engine.dispose())
            logger.info("数据库连接已关闭")
        except Exception as e:
            logger.error(f"关闭数据库连接失败: {e}")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close() 
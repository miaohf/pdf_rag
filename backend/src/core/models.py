"""
SQLAlchemy ORM模型

定义数据库表结构和关系。
"""

from sqlalchemy import Column, String, Text, Integer, BigInteger, TIMESTAMP, ForeignKey, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from sqlalchemy.dialects.postgresql import UUID
from pgvector.sqlalchemy import Vector
import uuid

Base = declarative_base()


class Document(Base):
    """文档表"""
    __tablename__ = 'documents'
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    filename = Column(String(255), nullable=False)
    file_path = Column(String(500))
    file_size = Column(BigInteger)
    file_hash = Column(String(64))
    title = Column(String(500))
    content_type = Column(String(100))
    page_count = Column(Integer, default=0)
    chunk_count = Column(Integer, default=0)
    status = Column(String(50), default='pending')
    upload_time = Column(TIMESTAMP, default=func.now())
    processed = Column(Integer, default=0)  # 0: 未处理, 1: 处理中, 2: 已处理, -1: 处理失败
    doc_metadata = Column('metadata', JSON, default=dict)
    created_time = Column(TIMESTAMP, default=func.now())
    updated_time = Column(TIMESTAMP, default=func.now(), onupdate=func.now())
    
    # 关系
    chunks = relationship("DocumentChunk", back_populates="document", cascade="all, delete-orphan")


class DocumentChunk(Base):
    """文档块表"""
    __tablename__ = 'document_chunks'
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    document_id = Column(String(36), ForeignKey('documents.id', ondelete='CASCADE'), nullable=False)
    content = Column(Text, nullable=False)  # 纯文本内容，用于向量检索
    original_content = Column(Text)  # 保留Markdown格式的原始内容，用于前端高亮匹配
    chunk_index = Column(Integer, nullable=False)
    start_char = Column(Integer)
    end_char = Column(Integer)
    parent_chunk_id = Column(String(36))  # 父文档片段ID（用于父子文档模式）
    chunk_metadata = Column('metadata', JSON, default=dict)
    created_time = Column(TIMESTAMP, default=func.now())
    
    # 关系
    document = relationship("Document", back_populates="chunks")
    vectors = relationship("DocumentVector", back_populates="chunk", cascade="all, delete-orphan")


class DocumentVector(Base):
    """文档向量表"""
    __tablename__ = 'document_vectors'
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    chunk_id = Column(String(36), ForeignKey('document_chunks.id', ondelete='CASCADE'), nullable=False)
    embedding = Column(Vector(768))  # 768维向量
    created_time = Column(TIMESTAMP, default=func.now())
    updated_time = Column(TIMESTAMP, default=func.now(), onupdate=func.now())
    
    # 关系
    chunk = relationship("DocumentChunk", back_populates="vectors")


class QueryHistory(Base):
    """查询历史表"""
    __tablename__ = 'query_history'
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    session_id = Column(String(36))
    user_id = Column(String(100))
    query_text = Column(Text, nullable=False)
    query_type = Column(String(50))
    response_text = Column(Text)
    confidence = Column(String(50))  # 增加长度以容纳浮点数字符串
    sources_count = Column(Integer, default=0)
    response_time = Column(String(50))  # 增加长度以容纳更长的时间字符串
    created_time = Column(TIMESTAMP, default=func.now())
    
    # 元数据
    query_metadata = Column('metadata', JSON, default=dict)


class Conversation(Base):
    """对话表"""
    __tablename__ = 'conversations'
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    session_id = Column(String(36), unique=True, nullable=False)
    user_id = Column(String(100))
    conversation_data = Column(Text)  # JSON格式的对话数据
    state = Column(String(20), default='active')
    created_time = Column(TIMESTAMP, default=func.now())
    last_activity = Column(TIMESTAMP, default=func.now(), onupdate=func.now())
    
    # 元数据
    conv_metadata = Column('metadata', JSON, default=dict) 
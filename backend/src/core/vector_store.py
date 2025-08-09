"""
向量存储

基于pgvector的向量存储和操作。
"""

import json
import uuid
from typing import List, Dict, Any, Optional, Tuple
from sqlalchemy import text
from datetime import datetime

from src.utils.config import Config
from src.utils.logger import get_logger
from src.utils.helpers import safe_json_dumps, safe_json_loads
from src.core.database import Database
from src.core.embeddings import EmbeddingModel

logger = get_logger(__name__)


class VectorStore:
    """向量存储"""
    
    def __init__(self, config: Config, database: Database, embedding_model: Optional[EmbeddingModel] = None):
        """
        初始化向量存储
        
        Args:
            config: 配置对象
            database: 数据库对象
            embedding_model: 嵌入模型（可选）
        """
        self.config = config
        self.db = database
        self.embedding_model = embedding_model or EmbeddingModel(config)
        self.retrieval_config = config.retrieval
        
        logger.info("向量存储初始化完成")
        
        # 确保向量扩展和表存在
        self._ensure_vector_extension()
        self._ensure_vector_table()
    
    def _ensure_vector_extension(self) -> None:
        """确保pgvector扩展已安装"""
        try:
            with self.db.get_session() as session:
                session.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
                logger.info("pgvector扩展检查完成")
        except Exception as e:
            logger.error(f"检查pgvector扩展失败: {e}")
    
    def _ensure_vector_table(self) -> None:
        """确保向量表存在"""
        try:
            with self.db.get_session() as session:
                # 获取嵌入向量维度
                dimension = self.embedding_model.get_embedding_dimension()
                
                # 创建向量表
                session.execute(text(f"""
                    CREATE TABLE IF NOT EXISTS document_vectors (
                        id VARCHAR(36) PRIMARY KEY,
                        chunk_id VARCHAR(36) REFERENCES document_chunks(id) ON DELETE CASCADE,
                        embedding VECTOR({dimension}),
                        created_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """))
                
                # 创建向量索引
                session.execute(text("""
                    CREATE INDEX IF NOT EXISTS idx_document_vectors_embedding 
                    ON document_vectors USING ivfflat (embedding vector_cosine_ops)
                    WITH (lists = 100)
                """))
                
                logger.info(f"向量表创建完成，维度: {dimension}")
                
        except Exception as e:
            logger.error(f"创建向量表失败: {e}")
    
    def add_vector(self, chunk_id: str, content: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        添加向量到存储
        
        Args:
            chunk_id: 文档片段ID
            content: 文档内容
            metadata: 元数据
            
        Returns:
            向量ID
        """
        try:
            # 生成嵌入向量
            embedding = self.embedding_model.embed_text(content)
            
            if not embedding:
                raise ValueError("生成嵌入向量失败")
            
            # 存储向量
            vector_id = str(uuid.uuid4())
            
            with self.db.get_session() as session:
                session.execute(
                    text("""
                        INSERT INTO document_vectors (id, chunk_id, embedding, created_time)
                        VALUES (:id, :chunk_id, :embedding, :created_time)
                    """),
                    {
                        "id": vector_id,
                        "chunk_id": chunk_id,
                        "embedding": embedding,
                        "created_time": datetime.utcnow()
                    }
                )
                
                logger.debug(f"向量添加成功: {vector_id}")
                return vector_id
                
        except Exception as e:
            logger.error(f"添加向量失败: {e}")
            raise
    
    def add_vectors_batch(self, chunks: List[Dict[str, Any]]) -> List[str]:
        """
        批量添加向量
        
        Args:
            chunks: 包含chunk_id和content的字典列表
            
        Returns:
            向量ID列表
        """
        if not chunks:
            return []
        
        logger.info(f"开始批量添加向量: {len(chunks)} 个")
        
        try:
            # 批量生成嵌入向量
            contents = [chunk['content'] for chunk in chunks]
            embeddings = self.embedding_model.embed_batch(contents)
            
            vector_ids = []
            
            with self.db.get_session() as session:
                for chunk, embedding in zip(chunks, embeddings):
                    if embedding and any(v != 0 for v in embedding):  # 检查非零向量
                        vector_id = str(uuid.uuid4())
                        
                        session.execute(
                            text("""
                                INSERT INTO document_vectors (id, chunk_id, embedding, created_time)
                                VALUES (:id, :chunk_id, :embedding, :created_time)
                            """),
                            {
                                "id": vector_id,
                                "chunk_id": chunk['chunk_id'],
                                "embedding": embedding,
                                "created_time": datetime.utcnow()
                            }
                        )
                        
                        vector_ids.append(vector_id)
                    else:
                        logger.warning(f"跳过无效向量: {chunk['chunk_id']}")
            
            logger.info(f"批量添加向量完成: {len(vector_ids)}/{len(chunks)}")
            return vector_ids
            
        except Exception as e:
            logger.error(f"批量添加向量失败: {e}")
            raise
    
    def search_similar(
        self, 
        query: str, 
        top_k: Optional[int] = None,
        similarity_threshold: Optional[float] = None,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        搜索相似向量（使用ORM）
        
        Args:
            query: 查询文本
            top_k: 返回数量
            similarity_threshold: 相似度阈值
            filter_metadata: 元数据过滤条件
            
        Returns:
            相似文档列表
        """
        top_k = top_k or self.retrieval_config.top_k
        similarity_threshold = similarity_threshold or self.retrieval_config.similarity_threshold
        
        try:
            # 生成查询向量
            query_embedding = self.embedding_model.embed_text(query)
            
            if not query_embedding:
                logger.error("生成查询向量失败")
                return []
            
            from src.core.models import DocumentVector, DocumentChunk, Document
            from sqlalchemy.orm import joinedload
            from sqlalchemy import func
            
            with self.db.get_session() as session:
                # 计算相似度表达式
                similarity_expr = 1 - DocumentVector.embedding.cosine_distance(query_embedding)
                
                # 构建ORM查询
                query_orm = session.query(
                    DocumentVector.id.label('vector_id'),
                    DocumentVector.chunk_id,
                    DocumentChunk.content,
                    DocumentChunk.chunk_metadata.label('chunk_metadata'),
                    Document.filename,
                    Document.file_path,
                    similarity_expr.label('similarity')
                ).join(
                    DocumentChunk, DocumentVector.chunk_id == DocumentChunk.id
                ).join(
                    Document, DocumentChunk.document_id == Document.id
                ).filter(
                    similarity_expr >= similarity_threshold
                ).order_by(
                    similarity_expr.desc()
                ).limit(top_k)
                
                # 执行查询
                results = []
                for row in query_orm.all():
                    chunk_metadata = row.chunk_metadata or {}
                    
                    results.append({
                        'vector_id': row.vector_id,
                        'chunk_id': row.chunk_id,
                        'content': row.content,
                        'metadata': chunk_metadata,
                        'filename': row.filename,
                        'file_path': row.file_path,
                        'similarity': float(row.similarity),
                        'source': f"{row.filename} (相似度: {row.similarity:.3f})"
                    })
                
                logger.info(f"ORM向量搜索完成: 找到 {len(results)} 个相似文档")
                return results
                
        except Exception as e:
            logger.error(f"ORM向量搜索失败: {e}")
            return []
    
    def search_by_vector(
        self, 
        embedding: List[float], 
        top_k: Optional[int] = None,
        similarity_threshold: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        通过向量搜索相似文档
        
        Args:
            embedding: 查询向量
            top_k: 返回数量
            similarity_threshold: 相似度阈值
            
        Returns:
            相似文档列表
        """
        top_k = top_k or self.retrieval_config.top_k
        similarity_threshold = similarity_threshold or self.retrieval_config.similarity_threshold
        
        try:
            with self.db.get_session() as session:
                result = session.execute(
                    text("""
                        SELECT 
                            dv.id as vector_id,
                            dv.chunk_id,
                            dc.content,
                            dc.metadata as chunk_metadata,
                            d.filename,
                            1 - (dv.embedding <=> :query_embedding) as similarity
                        FROM document_vectors dv
                        JOIN document_chunks dc ON dv.chunk_id = dc.id
                        JOIN documents d ON dc.document_id = d.id
                        WHERE 1 - (dv.embedding <=> :query_embedding) >= :threshold
                        ORDER BY similarity DESC
                        LIMIT :limit
                    """),
                    {
                        "query_embedding": embedding,
                        "threshold": similarity_threshold,
                        "limit": top_k
                    }
                )
                
                results = []
                for row in result.fetchall():
                    chunk_metadata = safe_json_loads(row[3], {})
                    
                    results.append({
                        'vector_id': row[0],
                        'chunk_id': row[1],
                        'content': row[2],
                        'metadata': chunk_metadata,
                        'filename': row[4],
                        'similarity': float(row[5])
                    })
                
                return results
                
        except Exception as e:
            logger.error(f"通过向量搜索失败: {e}")
            return []
    
    def update_vector(self, vector_id: str, content: str) -> bool:
        """
        更新向量
        
        Args:
            vector_id: 向量ID
            content: 新的内容
            
        Returns:
            是否更新成功
        """
        try:
            # 生成新的嵌入向量
            embedding = self.embedding_model.embed_text(content)
            
            if not embedding:
                return False
            
            with self.db.get_session() as session:
                result = session.execute(
                    text("""
                        UPDATE document_vectors 
                        SET embedding = :embedding, updated_time = :updated_time
                        WHERE id = :id
                    """),
                    {
                        "id": vector_id,
                        "embedding": embedding,
                        "updated_time": datetime.utcnow()
                    }
                )
                
                success = result.rowcount > 0
                if success:
                    logger.info(f"向量更新成功: {vector_id}")
                else:
                    logger.warning(f"向量不存在: {vector_id}")
                
                return success
                
        except Exception as e:
            logger.error(f"更新向量失败: {e}")
            return False
    
    def delete_vector(self, vector_id: str) -> bool:
        """
        删除向量
        
        Args:
            vector_id: 向量ID
            
        Returns:
            是否删除成功
        """
        try:
            with self.db.get_session() as session:
                result = session.execute(
                    text("DELETE FROM document_vectors WHERE id = :id"),
                    {"id": vector_id}
                )
                
                success = result.rowcount > 0
                if success:
                    logger.info(f"向量删除成功: {vector_id}")
                else:
                    logger.warning(f"向量不存在: {vector_id}")
                
                return success
                
        except Exception as e:
            logger.error(f"删除向量失败: {e}")
            return False
    
    def delete_vectors_by_chunk(self, chunk_id: str) -> int:
        """
        删除文档片段的所有向量
        
        Args:
            chunk_id: 片段ID
            
        Returns:
            删除的向量数量
        """
        try:
            with self.db.get_session() as session:
                result = session.execute(
                    text("DELETE FROM document_vectors WHERE chunk_id = :chunk_id"),
                    {"chunk_id": chunk_id}
                )
                
                deleted_count = result.rowcount
                logger.info(f"删除片段向量: {chunk_id}, 数量: {deleted_count}")
                return deleted_count
                
        except Exception as e:
            logger.error(f"删除片段向量失败: {e}")
            return 0
    
    def get_vector_stats(self) -> Dict[str, Any]:
        """
        获取向量存储统计信息
        
        Returns:
            统计信息字典
        """
        try:
            with self.db.get_session() as session:
                # 向量总数
                result = session.execute(text("SELECT COUNT(*) FROM document_vectors"))
                total_vectors = result.scalar()
                
                # 相关的文档数
                result = session.execute(text("""
                    SELECT COUNT(DISTINCT dc.document_id) 
                    FROM document_vectors dv
                    JOIN document_chunks dc ON dv.chunk_id = dc.id
                """))
                vectorized_documents = result.scalar()
                
                # 平均相似度（如果有查询历史）
                embedding_stats = self.embedding_model.get_cache_stats()
                
                return {
                    'total_vectors': total_vectors or 0,
                    'vectorized_documents': vectorized_documents or 0,
                    'embedding_dimension': self.embedding_model.get_embedding_dimension(),
                    'cache_stats': embedding_stats
                }
                
        except Exception as e:
            logger.error(f"获取向量统计失败: {e}")
            return {}
    
    def rebuild_index(self) -> bool:
        """
        重建向量索引
        
        Returns:
            是否重建成功
        """
        try:
            with self.db.get_session() as session:
                # 删除旧索引
                session.execute(text("DROP INDEX IF EXISTS idx_document_vectors_embedding"))
                
                # 重建索引
                session.execute(text("""
                    CREATE INDEX idx_document_vectors_embedding 
                    ON document_vectors USING ivfflat (embedding vector_cosine_ops)
                    WITH (lists = 100)
                """))
                
                logger.info("向量索引重建完成")
                return True
                
        except Exception as e:
            logger.error(f"重建向量索引失败: {e}")
            return False 
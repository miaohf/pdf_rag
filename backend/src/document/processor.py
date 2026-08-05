"""
文档处理器

负责文档的完整处理流程：加载、分片、向量化、存储。
"""

import uuid
import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Union

from src.utils.config import Config
from src.utils.logger import get_logger
from src.utils.helpers import safe_json_dumps, format_file_size
from src.core.database import Database
from src.core.models import Document, DocumentChunk, DocumentVector
from src.core.embeddings import EmbeddingModel
from src.document.loader import DocumentLoader
from src.document.chunker import DocumentChunker, Chunk

logger = get_logger(__name__)


class DocumentProcessor:
    """文档处理器"""
    
    def __init__(self, config: Config, database: Optional[Database] = None, enable_vectorization: bool = True, force_reprocess: bool = False):
        """
        初始化文档处理器
        
        Args:
            config: 配置对象
            database: 数据库对象，如果为None则创建新实例
            enable_vectorization: 是否启用向量化
            force_reprocess: 是否强制重新处理已存在的文档
        """
        self.config = config
        self.db = database or Database(config)
        self.loader = DocumentLoader()
        self.chunker = DocumentChunker(config)
        self.enable_vectorization = enable_vectorization
        self.force_reprocess = force_reprocess
        
        # 初始化嵌入模型（如果启用向量化）
        if self.enable_vectorization:
            try:
                self.embedding_model = EmbeddingModel(config)
                logger.info("嵌入模型初始化完成")
            except Exception as e:
                logger.warning(f"嵌入模型初始化失败，将跳过向量化: {e}")
                self.enable_vectorization = False
                self.embedding_model = None
        else:
            self.embedding_model = None
        
        logger.info("文档处理器初始化完成")
    
    def process_file(self, file_path: Union[str, Path]) -> List[str]:
        """
        处理单个文件
        
        Args:
            file_path: 文件路径
            
        Returns:
            生成的文档片段ID列表
        """
        file_path = Path(file_path)
        logger.info(f"开始处理文件: {file_path}")
        
        try:
            # 1. 加载文档
            document = self.loader.load(file_path)
            logger.info(f"文档加载完成: {format_file_size(document['metadata']['file_size'])}")
            
            # 2. 检查文件是否已处理过
            document_id, is_existing = self._get_or_create_document_record(document)
            
            # 如果文档已存在，检查是否需要重新处理
            if is_existing:
                with self.db.get_session() as session:
                    existing_doc = session.query(Document).filter(Document.id == document_id).first()
                    
                    # 如果强制重新处理，直接清理旧数据
                    if self.force_reprocess:
                        logger.info(f"强制重新处理文档，清理旧数据: {document['metadata'].get('filename', 'unknown')}")
                    else:
                        # 如果文档已完全处理且有向量，跳过处理
                        if existing_doc and existing_doc.processed == 2:
                            chunk_count = session.query(DocumentChunk).filter(
                                DocumentChunk.document_id == document_id
                            ).count()
                            
                            vector_count = session.query(DocumentVector).join(
                                DocumentChunk, DocumentVector.chunk_id == DocumentChunk.id
                            ).filter(DocumentChunk.document_id == document_id).count()
                            
                            if chunk_count > 0 and chunk_count == vector_count:
                                logger.info(f"文档已完全处理，跳过: {document['metadata'].get('filename', 'unknown')}")
                                # 返回现有的chunk_ids
                                existing_chunks = session.query(DocumentChunk.id).filter(
                                    DocumentChunk.document_id == document_id
                                ).all()
                                return [chunk.id for chunk in existing_chunks]
                        
                        # 如果文档存在但未完全处理，清理旧数据重新处理
                        logger.info(f"文档存在但未完全处理，清理旧数据重新处理: {document['metadata'].get('filename', 'unknown')}")
                    
                    # 删除旧的向量和片段
                    session.query(DocumentVector).filter(
                        DocumentVector.chunk_id.in_(
                            session.query(DocumentChunk.id).filter(
                                DocumentChunk.document_id == document_id
                            )
                        )
                    ).delete(synchronize_session=False)
                    
                    session.query(DocumentChunk).filter(
                        DocumentChunk.document_id == document_id
                    ).delete()
                    
                    session.commit()
            
            # 3. 分片处理 - 使用父子分片策略
            if self.chunker.use_hierarchical_chunking:
                logger.info("使用父子分片策略处理文档")
                hierarchical_chunks = self.chunker.create_parent_child_chunks(
                    document['content'],
                    document['content']
                )
                
                # 主要使用子分片进行检索，父分片用于上下文扩展
                chunks = hierarchical_chunks['child_chunks']
                parent_chunks = hierarchical_chunks['parent_chunks']
                
                # 记录父子关系信息
                logger.info(f"父子分片完成: {len(parent_chunks)} 个父分片, {len(chunks)} 个子分片")
            else:
                logger.info("使用传统单层分片策略处理文档")
                chunks = self.chunker.smart_chunk(document['content'])
            
            # 注意：现在content字段已经保存原始格式的内容，无需额外处理
            
            logger.info(f"文档分片完成: {len(chunks)} 个片段")
            
            # 4. 存储片段到数据库（包含向量化）
            if self.chunker.use_hierarchical_chunking:
                # 对于父子分片，我们主要存储子分片用于检索，但也记录父分片信息
                chunk_ids = self._store_chunks(document_id, chunks, document['metadata'])
                # 注意：父分片信息已经通过parent_chunk_id字段记录在子分片中
            else:
                chunk_ids = self._store_chunks(document_id, chunks, document['metadata'])
            
            # 5. 更新文档状态
            if self.enable_vectorization:
                # 检查向量化是否完成
                with self.db.get_session() as session:
                    vector_count = session.query(DocumentVector).join(
                        DocumentChunk, DocumentVector.chunk_id == DocumentChunk.id
                    ).filter(DocumentChunk.document_id == document_id).count()
                    
                    if vector_count == len(chunks):
                        # 所有片段都已向量化，设置为已索引
                        status = 2  # 已处理
                        status_name = "已处理并索引"
                    else:
                        # 部分向量化或向量化失败
                        status = 1  # 处理中
                        status_name = "已处理，部分索引"
            else:
                # 未启用向量化，仅标记为已处理
                status = 2
                status_name = "已处理"
            
            self._update_document_status(document_id, len(chunks), status=status)
            logger.info(f"文档状态: {status_name}")
            
            logger.info(f"文件处理完成: {file_path}, 生成 {len(chunk_ids)} 个片段")
            return chunk_ids
            
        except Exception as e:
            logger.error(f"处理文件失败 {file_path}: {e}")
            # 更新文档状态为失败
            try:
                document_id = self._get_or_create_document_record({'metadata': {'file_path': str(file_path)}})
                self._update_document_status(document_id, 0, status=-1)  # 处理失败
            except:
                pass
            raise
    
    def process_directory(self, directory_path: Union[str, Path], recursive: bool = True) -> Dict[str, Any]:
        """
        处理目录中的所有文档
        
        Args:
            directory_path: 目录路径
            recursive: 是否递归处理子目录
            
        Returns:
            处理结果统计
        """
        directory_path = Path(directory_path)
        logger.info(f"开始处理目录: {directory_path}")
        
        if not directory_path.exists() or not directory_path.is_dir():
            raise ValueError(f"目录不存在或不是目录: {directory_path}")
        
        # 获取支持的文件
        supported_extensions = set(self.loader.get_supported_extensions())
        files = []
        
        if recursive:
            for ext in supported_extensions:
                files.extend(directory_path.rglob(f"*{ext}"))
        else:
            for ext in supported_extensions:
                files.extend(directory_path.glob(f"*{ext}"))
        
        logger.info(f"找到 {len(files)} 个支持的文档文件")
        
        # 处理统计
        stats = {
            'total_files': len(files),
            'processed_files': 0,
            'failed_files': 0,
            'total_chunks': 0,
            'file_results': []
        }
        
        # 逐个处理文件
        for file_path in files:
            try:
                chunk_ids = self.process_file(file_path)
                stats['processed_files'] += 1
                stats['total_chunks'] += len(chunk_ids)
                stats['file_results'].append({
                    'file_path': str(file_path),
                    'status': 'success',
                    'chunk_count': len(chunk_ids),
                    'chunk_ids': chunk_ids
                })
                logger.info(f"✅ {file_path.name}: {len(chunk_ids)} 个片段")
                
            except Exception as e:
                stats['failed_files'] += 1
                stats['file_results'].append({
                    'file_path': str(file_path),
                    'status': 'failed',
                    'error': str(e)
                })
                logger.error(f"❌ {file_path.name}: {e}")
        
        logger.info(f"目录处理完成: 成功 {stats['processed_files']}/{stats['total_files']} 个文件, 生成 {stats['total_chunks']} 个片段")
        return stats
    
    def _get_or_create_document_record(self, document: Dict[str, Any]) -> tuple[str, bool]:
        """
        获取或创建文档记录
        
        Args:
            document: 文档数据
            
        Returns:
            (文档ID, 是否已存在)
        """
        metadata = document['metadata']
        file_hash = metadata.get('file_hash', '')
        
        try:
            with self.db.get_session() as session:
                # 检查是否已存在
                existing = session.query(Document).filter(Document.file_hash == file_hash).first()
                
                if existing:
                    logger.info(f"文档已存在: {metadata.get('filename', 'unknown')}")
                    return existing.id, True
                
                # 创建新记录
                new_document = Document(
                    filename=metadata.get('filename', ''),
                    file_path=metadata.get('file_path', ''),
                    file_size=metadata.get('file_size', 0),
                    file_hash=file_hash,
                    title=metadata.get('title', ''),
                    content_type=metadata.get('content_type', ''),
                    upload_time=datetime.utcnow(),
                    processed=1,  # 处理中
                    doc_metadata=metadata
                )
                
                session.add(new_document)
                session.commit()
                
                logger.info(f"创建文档记录: {metadata.get('filename', 'unknown')}")
                return new_document.id, False
                
        except Exception as e:
            logger.error(f"创建文档记录失败: {e}")
            raise
    
    def _store_chunks(self, document_id: str, chunks: List[Chunk], document_metadata: Dict[str, Any]) -> List[str]:
        """
        存储文档片段到数据库并生成向量
        
        Args:
            document_id: 文档ID
            chunks: 文档片段列表
            document_metadata: 文档元数据
            
        Returns:
            片段ID列表
        """
        chunk_ids = []
        vectorized_count = 0
        
        try:
            with self.db.get_session() as session:
                for i, chunk in enumerate(chunks):
                    # 准备片段元数据
                    chunk_metadata = {
                        **chunk.metadata,
                        'document_filename': document_metadata.get('filename', ''),
                        'document_type': document_metadata.get('content_type', ''),
                        'created_time': datetime.utcnow().isoformat()
                    }
                    
                    # 创建文档片段记录
                    new_chunk = DocumentChunk(
                        document_id=document_id,
                        content=chunk.content,
                        chunk_index=chunk.index,
                        start_char=chunk.start_pos,
                        end_char=chunk.end_pos,
                        parent_chunk_id=chunk.parent_chunk_id,  # 设置父分片ID
                        chunk_metadata=chunk_metadata
                    )
                    
                    session.add(new_chunk)
                    session.flush()  # 确保获得chunk ID
                    chunk_ids.append(new_chunk.id)
                    
                    # 如果启用向量化，立即生成向量
                    if self.enable_vectorization and self.embedding_model:
                        try:
                            logger.debug(f"正在向量化片段 {i+1}/{len(chunks)}")
                            embedding = self.embedding_model.embed_text(chunk.content)
                            
                            if embedding and len(embedding) > 0:
                                # 创建向量记录
                                vector = DocumentVector(
                                    chunk_id=new_chunk.id,
                                    embedding=embedding
                                )
                                session.add(vector)
                                vectorized_count += 1
                                logger.debug(f"片段 {i+1} 向量化成功 (维度: {len(embedding)})")
                            else:
                                logger.warning(f"片段 {i+1} 向量化失败")
                                
                        except Exception as e:
                            logger.error(f"片段 {i+1} 向量化出错: {e}")
                            # 向量化失败不影响文档片段存储
                            continue
                
                session.commit()
                
                if self.enable_vectorization:
                    logger.info(f"存储 {len(chunk_ids)} 个文档片段，向量化 {vectorized_count} 个")
                else:
                    logger.info(f"存储 {len(chunk_ids)} 个文档片段")
                    
                return chunk_ids
                
        except Exception as e:
            logger.error(f"存储文档片段失败: {e}")
            raise
    
    def _update_document_status(self, document_id: str, chunk_count: int, status: int):
        """
        更新文档处理状态
        
        Args:
            document_id: 文档ID
            chunk_count: 片段数量
            status: 状态 (0: 未处理, 1: 处理中, 2: 已处理, -1: 失败)
        """
        try:
            with self.db.get_session() as session:
                document = session.query(Document).filter(Document.id == document_id).first()
                if document:
                    document.processed = status
                    document.chunk_count = chunk_count
                    document.updated_time = datetime.utcnow()
                    session.commit()
                    
                    status_text = {0: "未处理", 1: "处理中", 2: "已处理", -1: "失败"}.get(status, "未知")
                    logger.info(f"更新文档状态: {document_id} -> {status_text}")
                else:
                    logger.error(f"文档不存在: {document_id}")
                
        except Exception as e:
            logger.error(f"更新文档状态失败: {e}")
    
    def get_document_info(self, document_id: str) -> Optional[Dict[str, Any]]:
        """
        获取文档信息
        
        Args:
            document_id: 文档ID
            
        Returns:
            文档信息或None
        """
        try:
            with self.db.get_session() as session:
                document = session.query(Document).filter(Document.id == document_id).first()
                
                if document:
                    return {
                        'id': document.id,
                        'filename': document.filename,
                        'file_path': document.file_path,
                        'file_size': document.file_size,
                        'content_type': document.content_type,
                        'upload_time': document.upload_time,
                        'processed': document.processed,
                        'chunk_count': document.chunk_count,
                        'metadata': document.doc_metadata or {}
                    }
                
                return None
                
        except Exception as e:
            logger.error(f"获取文档信息失败: {e}")
            return None
    
    def list_documents(self, limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
        """
        列出文档
        
        Args:
            limit: 限制数量
            offset: 偏移量
            
        Returns:
            文档列表
        """
        try:
            with self.db.get_session() as session:
                documents_query = session.query(Document).order_by(Document.upload_time.desc()).limit(limit).offset(offset)
                
                documents = []
                for doc in documents_query:
                    documents.append({
                        'id': doc.id,
                        'filename': doc.filename,
                        'file_size': doc.file_size,
                        'content_type': doc.content_type,
                        'upload_time': doc.upload_time,
                        'processed': doc.processed,
                        'chunk_count': doc.chunk_count
                    })
                
                return documents
                
        except Exception as e:
            logger.error(f"列出文档失败: {e}")
            return []
    
    def delete_document(self, document_id: str) -> bool:
        """
        删除文档及其所有片段
        
        Args:
            document_id: 文档ID
            
        Returns:
            是否删除成功
        """
        try:
            with self.db.get_session() as session:
                # 通过ORM删除文档（会自动级联删除相关的文档片段）
                document = session.query(Document).filter(Document.id == document_id).first()
                
                if document:
                    session.delete(document)
                    session.commit()
                    logger.info(f"删除文档: {document_id}")
                    return True
                else:
                    logger.warning(f"文档不存在: {document_id}")
                    return False
                
        except Exception as e:
            logger.error(f"删除文档失败: {e}")
            return False 

 
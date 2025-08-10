#!/usr/bin/env python3
"""
父子分片两阶段检索器

实现基于父子分片关系的高效检索策略。
"""

from typing import List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass

from src.utils.config import Config
from src.utils.logger import get_logger
from src.core.database import Database
from src.core.embeddings import EmbeddingModel
from src.core.vector_store import VectorStore

logger = get_logger(__name__)


@dataclass
class HierarchicalResult:
    """分层检索结果"""
    child_chunks: List[Dict[str, Any]]      # 精准匹配的子分片
    parent_contexts: List[Dict[str, Any]]   # 对应的父分片上下文
    total_found: int                        # 总发现数
    child_count: int                        # 子分片数量
    parent_count: int                       # 父分片数量
    retrieval_strategy: str                 # 检索策略


class HierarchicalRetriever:
    """父子分片两阶段检索器"""
    
    def __init__(
        self, 
        config: Config, 
        database: Database, 
        vector_store: VectorStore,
        embedding_model: Optional[EmbeddingModel] = None
    ):
        """
        初始化分层检索器
        
        Args:
            config: 配置对象
            database: 数据库对象
            vector_store: 向量存储
            embedding_model: 嵌入模型
        """
        self.config = config
        self.db = database
        self.vector_store = vector_store
        self.embedding_model = embedding_model or EmbeddingModel(config)
        
        # 检索配置
        self.child_top_k = config.retrieval.top_k  # 子分片检索数量
        self.parent_expansion_ratio = 1.5  # 父分片扩展比例
        self.context_window = 2  # 相邻分片窗口大小
        
        logger.info("父子分片两阶段检索器初始化完成")
    
    def hierarchical_search(
        self,
        query: str,
        strategy: str = "child_to_parent",
        top_k: Optional[int] = None,
        similarity_threshold: Optional[float] = None,
        include_siblings: bool = True
    ) -> HierarchicalResult:
        """
        执行分层检索
        
        Args:
            query: 查询文本
            strategy: 检索策略 ("child_to_parent", "parent_to_child", "hybrid")
            top_k: 返回数量
            similarity_threshold: 相似度阈值
            include_siblings: 是否包含兄弟分片
            
        Returns:
            分层检索结果
        """
        top_k = top_k or self.child_top_k
        
        logger.info(f"开始分层检索: 策略={strategy}, top_k={top_k}, 查询='{query}'")
        
        if strategy == "child_to_parent":
            return self._child_to_parent_search(query, top_k, similarity_threshold, include_siblings)
        elif strategy == "parent_to_child":
            return self._parent_to_child_search(query, top_k, similarity_threshold)
        elif strategy == "hybrid":
            return self._hybrid_search(query, top_k, similarity_threshold, include_siblings)
        else:
            raise ValueError(f"不支持的检索策略: {strategy}")
    
    def _child_to_parent_search(
        self,
        query: str,
        top_k: int,
        similarity_threshold: Optional[float],
        include_siblings: bool
    ) -> HierarchicalResult:
        """
        子分片到父分片检索策略
        1. 在子分片中进行精准检索
        2. 获取对应的父分片提供上下文
        3. 可选地包含兄弟分片
        """
        # 第一阶段：子分片精准检索
        child_results = self.vector_store.search_similar(
            query=query,
            top_k=top_k,
            similarity_threshold=similarity_threshold
        )
        
        if not child_results:
            return HierarchicalResult(
                child_chunks=[],
                parent_contexts=[],
                total_found=0,
                child_count=0,
                parent_count=0,
                retrieval_strategy="child_to_parent"
            )
        
        logger.info(f"第一阶段：找到 {len(child_results)} 个相关子分片")
        
        # 第二阶段：获取父分片上下文
        parent_contexts = self._get_parent_contexts(child_results, include_siblings)
        
        logger.info(f"第二阶段：获取 {len(parent_contexts)} 个父分片上下文")
        
        # 合并并去重
        final_chunks = self._merge_and_deduplicate(child_results, parent_contexts)
        
        return HierarchicalResult(
            child_chunks=child_results,
            parent_contexts=parent_contexts,
            total_found=len(final_chunks),
            child_count=len(child_results),
            parent_count=len(parent_contexts),
            retrieval_strategy="child_to_parent"
        )
    
    def _parent_to_child_search(
        self,
        query: str,
        top_k: int,
        similarity_threshold: Optional[float]
    ) -> HierarchicalResult:
        """
        父分片到子分片检索策略
        1. 首先在更大的语义单元（父分片概念）中检索
        2. 然后获取相关的精确子分片
        """
        # 注意：当前所有分片都是子分片，这个策略需要父分片数据
        # 这里提供一个概念实现，实际需要父分片在数据库中
        
        logger.warning("父分片到子分片策略需要父分片数据，当前使用子分片检索")
        child_results = self.vector_store.search_similar(
            query=query,
            top_k=top_k,
            similarity_threshold=similarity_threshold
        )
        
        return HierarchicalResult(
            child_chunks=child_results,
            parent_contexts=[],
            total_found=len(child_results),
            child_count=len(child_results),
            parent_count=0,
            retrieval_strategy="parent_to_child"
        )
    
    def _hybrid_search(
        self,
        query: str,
        top_k: int,
        similarity_threshold: Optional[float],
        include_siblings: bool
    ) -> HierarchicalResult:
        """
        混合检索策略
        结合子分片精准检索和父分片上下文扩展
        """
        # 执行子分片到父分片检索
        result = self._child_to_parent_search(query, top_k, similarity_threshold, include_siblings)
        
        # 可以在这里添加更多的增强逻辑
        # 例如：相邻分片检索、关键词扩展等
        
        result.retrieval_strategy = "hybrid"
        return result
    
    def _get_parent_contexts(
        self,
        child_results: List[Dict[str, Any]],
        include_siblings: bool = True
    ) -> List[Dict[str, Any]]:
        """
        获取父分片上下文
        
        Args:
            child_results: 子分片检索结果
            include_siblings: 是否包含兄弟分片
            
        Returns:
            父分片上下文列表
        """
        from src.core.models import DocumentChunk
        
        # 提取父分片ID
        parent_ids = set()
        for child in child_results:
            chunk_id = child.get('chunk_id')
            if chunk_id:
                with self.db.get_session() as session:
                    chunk = session.query(DocumentChunk).filter(
                        DocumentChunk.id == chunk_id
                    ).first()
                    
                    if chunk and chunk.parent_chunk_id:
                        parent_ids.add(chunk.parent_chunk_id)
        
        if not parent_ids:
            logger.info("未找到父分片ID，可能数据中没有父子关系")
            return []
        
        parent_contexts = []
        
        with self.db.get_session() as session:
            for parent_id in parent_ids:
                # 获取该父分片的所有子分片
                child_chunks = session.query(DocumentChunk).filter(
                    DocumentChunk.parent_chunk_id == parent_id
                ).order_by(DocumentChunk.chunk_index).all()
                
                if child_chunks:
                    # 重构父分片内容（合并所有子分片）
                    parent_content = self._reconstruct_parent_content(child_chunks)
                    
                    parent_context = {
                        'parent_id': parent_id,
                        'content': parent_content,
                        'chunk_count': len(child_chunks),
                        'chunk_ids': [chunk.id for chunk in child_chunks],
                        'metadata': {
                            'type': 'reconstructed_parent',
                            'child_count': len(child_chunks),
                            'parent_id': parent_id
                        },
                        'source': f"父分片 {parent_id} (包含 {len(child_chunks)} 个子分片)"
                    }
                    
                    parent_contexts.append(parent_context)
                    
                    if include_siblings:
                        # 添加兄弟分片信息到元数据
                        parent_context['sibling_chunks'] = [
                            {
                                'chunk_id': chunk.id,
                                'content': chunk.content[:100] + "...",
                                'index': chunk.chunk_index
                            }
                            for chunk in child_chunks
                        ]
        
        return parent_contexts
    
    def _reconstruct_parent_content(self, child_chunks: List) -> str:
        """
        重构父分片内容
        
        Args:
            child_chunks: 子分片列表
            
        Returns:
            重构的父分片内容
        """
        # 按索引排序
        sorted_chunks = sorted(child_chunks, key=lambda x: x.chunk_index)
        
        # 合并内容，去除重复
        contents = []
        prev_content_end = ""
        
        for chunk in sorted_chunks:
            content = chunk.content
            
            # 简单的重叠去除
            if prev_content_end and content.startswith(prev_content_end[-50:]):
                # 如果当前内容开头与前一个内容结尾有重叠，去除重叠部分
                overlap_pos = content.find(prev_content_end[-50:])
                if overlap_pos >= 0:
                    content = content[overlap_pos + len(prev_content_end[-50:]):]
            
            contents.append(content)
            prev_content_end = content
        
        return " ".join(contents)
    
    def _merge_and_deduplicate(
        self,
        child_results: List[Dict[str, Any]],
        parent_contexts: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        合并并去重结果
        
        Args:
            child_results: 子分片结果
            parent_contexts: 父分片上下文
            
        Returns:
            合并后的结果列表
        """
        merged_results = []
        seen_ids = set()
        
        # 优先保留子分片（更精确）
        for child in child_results:
            chunk_id = child.get('chunk_id')
            if chunk_id not in seen_ids:
                child['result_type'] = 'child_chunk'
                merged_results.append(child)
                seen_ids.add(chunk_id)
        
        # 添加父分片上下文（去重）
        for parent in parent_contexts:
            parent_id = parent.get('parent_id')
            if parent_id not in seen_ids:
                parent['result_type'] = 'parent_context'
                merged_results.append(parent)
                seen_ids.add(parent_id)
        
        return merged_results
    
    def get_enhanced_context(self, hierarchical_result: HierarchicalResult) -> str:
        """
        获取增强上下文用于LLM生成
        
        Args:
            hierarchical_result: 分层检索结果
            
        Returns:
            增强上下文字符串
        """
        context_parts = []
        
        # 添加子分片（精确匹配）
        if hierarchical_result.child_chunks:
            context_parts.append("=== 精确匹配内容 ===")
            for i, child in enumerate(hierarchical_result.child_chunks, 1):
                context_parts.append(f"匹配片段 {i}:")
                context_parts.append(child.get('content', ''))
                context_parts.append("")
        
        # 添加父分片上下文（完整语义）
        if hierarchical_result.parent_contexts:
            context_parts.append("=== 完整上下文 ===")
            for i, parent in enumerate(hierarchical_result.parent_contexts, 1):
                context_parts.append(f"上下文 {i} (包含 {parent.get('chunk_count', 0)} 个子分片):")
                context_parts.append(parent.get('content', ''))
                context_parts.append("")
        
        return "\n".join(context_parts) 
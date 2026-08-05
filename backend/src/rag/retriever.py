"""向量检索器

整合父子分片检索、向量检索与基础统计，提供具备去重与元信息的统一输出。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class RetrievalResult:
    """单次检索的结果聚合"""

    query: str
    documents: List[Dict[str, Any]]
    child_chunks: List[Dict[str, Any]]
    parent_contexts: List[Dict[str, Any]]
    stats: Dict[str, Any]


class VectorRetriever:
    """结合层次检索与向量检索的统一检索器。"""

    def __init__(
        self,
        config,
        vector_store,
        hierarchical_retriever,
    ) -> None:
        self.config = config
        self.vector_store = vector_store
        self.hierarchical_retriever = hierarchical_retriever

        self.default_strategy = "hybrid"
        self.default_top_k = config.retrieval.top_k
        self.default_threshold = config.retrieval.similarity_threshold

        logger.info("向量检索器初始化完成")

    def retrieve(
        self,
        query: str,
        *,
        top_k: Optional[int] = None,
        similarity_threshold: Optional[float] = None,
        strategy: Optional[str] = None,
        include_siblings: bool = True,
    ) -> RetrievalResult:
        """执行一次标准检索，返回层次信息及聚合统计。"""

        top_k = top_k or self.default_top_k
        similarity_threshold = (
            similarity_threshold if similarity_threshold is not None else self.default_threshold
        )
        strategy = strategy or self.default_strategy

        hierarchical = self.hierarchical_retriever.hierarchical_search(
            query=query,
            strategy=strategy,
            top_k=top_k,
            similarity_threshold=similarity_threshold,
            include_siblings=include_siblings,
        )

        documents = self._merge_results(hierarchical.child_chunks, hierarchical.parent_contexts)

        if not documents:
            # 回退到纯向量检索，确保至少有结果
            fallback = self.vector_store.search_similar(
                query=query,
                top_k=top_k,
                similarity_threshold=similarity_threshold,
            )
            documents = self._merge_results(fallback, [])
            child_chunks = fallback
            parent_contexts = []
        else:
            child_chunks = hierarchical.child_chunks
            parent_contexts = hierarchical.parent_contexts

        stats = {
            "query": query,
            "strategy": strategy,
            "requested_top_k": top_k,
            "similarity_threshold": similarity_threshold,
            "child_count": len(child_chunks),
            "parent_count": len(parent_contexts),
            "document_count": len(documents),
        }

        return RetrievalResult(
            query=query,
            documents=documents,
            child_chunks=child_chunks,
            parent_contexts=parent_contexts,
            stats=stats,
        )

    def retrieve_for_queries(
        self,
        queries: List[str],
        *,
        top_k: Optional[int] = None,
        similarity_threshold: Optional[float] = None,
        deduplicate: bool = True,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        针对多个查询执行检索，返回合并后的文档列表与统计信息列表。
        """

        aggregated_docs: Dict[str, Dict[str, Any]] = {}
        stats: List[Dict[str, Any]] = []

        for q in queries:
            result = self.retrieve(
                q,
                top_k=top_k,
                similarity_threshold=similarity_threshold,
            )
            stats.append(result.stats)

            if not deduplicate:
                for doc in result.documents:
                    aggregated_docs[f"{doc.get('chunk_id')}-{len(aggregated_docs)}"] = doc
                continue

            for doc in result.documents:
                chunk_id = doc.get("chunk_id") or doc.get("parent_id")
                if not chunk_id:
                    chunk_id = f"anon_{len(aggregated_docs)}"

                existing = aggregated_docs.get(chunk_id)
                if existing is None or doc.get("similarity", 0.0) > existing.get("similarity", 0.0):
                    aggregated_docs[chunk_id] = doc

        ordered_docs = sorted(
            aggregated_docs.values(),
            key=lambda x: x.get("similarity", 0.0),
            reverse=True,
        )

        return ordered_docs, stats

    @staticmethod
    def _merge_results(
        child_chunks: List[Dict[str, Any]],
        parent_contexts: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """合并并去重子/父分片。"""

        merged: Dict[str, Dict[str, Any]] = {}

        for child in child_chunks or []:
            chunk_id = child.get("chunk_id")
            if not chunk_id:
                continue
            merged[chunk_id] = child

        for parent in parent_contexts or []:
            chunk_id = parent.get("chunk_id") or parent.get("parent_id")
            if not chunk_id:
                continue
            if chunk_id not in merged:
                merged[chunk_id] = parent
            else:
                # 选取更高的相似度，保留父分片补充元数据
                existing = merged[chunk_id]
                if parent.get("similarity", 0.0) > existing.get("similarity", 0.0):
                    merged[chunk_id] = parent

        ordered = sorted(
            merged.values(),
            key=lambda x: x.get("similarity", 0.0),
            reverse=True,
        )
        return ordered

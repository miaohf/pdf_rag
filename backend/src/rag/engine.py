"""统一 RAG 引擎

负责检索、查询扩展、重排序与答案生成的完整流水线。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from src.utils.config import Config
from src.utils.logger import get_logger
from src.core.llm import LLMClient
from src.rag.retriever import VectorRetriever
from src.rag.query_expander import QueryExpander
from src.rag.intelligent_expander import IntelligentExpander
from src.rag.prf_expander import LLMPRFExpander
from src.rag.chunk_scorer import ChunkScorer
from src.rag.content_filter import HybridContentFilter

logger = get_logger(__name__)


@dataclass
class EngineOutput:
    """统一引擎的执行结果"""

    question: str
    answer: str
    sources: List[Dict[str, Any]]
    retrieved_documents: List[Dict[str, Any]]
    selected_chunks: List[Dict[str, Any]]
    query_traces: List[Dict[str, Any]]
    expansion_log: Dict[str, List[str]]
    response_time: float
    confidence: float
    llm_metadata: Dict[str, Any] = field(default_factory=dict)
    fallback: bool = False


class UnifiedRAGEngine:
    """统一的检索增强生成引擎。"""

    def __init__(
        self,
        config: Config,
        vector_retriever: VectorRetriever,
        llm_client: LLMClient,
        chunk_scorer: ChunkScorer,
        content_filter: HybridContentFilter,
        *,
        query_expander: Optional[QueryExpander] = None,
        intelligent_expander: Optional[IntelligentExpander] = None,
        prf_expander: Optional[LLMPRFExpander] = None,
    ) -> None:
        self.config = config
        self.vector_retriever = vector_retriever
        self.llm_client = llm_client
        self.chunk_scorer = chunk_scorer
        self.content_filter = content_filter
        self.query_expander = query_expander or QueryExpander(config)
        self.intelligent_expander = intelligent_expander or IntelligentExpander(config)
        self.prf_expander = prf_expander

        self.default_max_chunks = 4
        self.system_prompt = (
            "你是一个专业的法律法规智能助手。请严格基于提供的文档片段回答用户问题，"
            "必须引用真实存在的内容，若缺少信息要明确说明。"
        )

        logger.info("统一RAG引擎初始化完成")

    def run(
        self,
        question: str,
        *,
        sub_questions: Optional[Sequence[str]] = None,
        top_k: Optional[int] = None,
        similarity_threshold: Optional[float] = None,
        max_chunks: Optional[int] = None,
        use_reranking: bool = True,
        enable_prf: bool = True,
        expansion_limit: int = 12,
    ) -> EngineOutput:
        """执行一次完整的 RAG 流程。"""

        start_time = time.time()
        queries = self._prepare_initial_queries(question, sub_questions)
        top_k = top_k or self.config.retrieval.top_k
        similarity_threshold = (
            similarity_threshold
            if similarity_threshold is not None
            else self.config.retrieval.similarity_threshold
        )
        max_chunks = max_chunks or self.default_max_chunks

        aggregated_docs: List[Dict[str, Any]] = []
        query_traces: List[Dict[str, Any]] = []
        seen_queries = set()

        # 1) 初次检索（原始问题 + 子问题）
        for q in queries:
            if q in seen_queries:
                continue
            seen_queries.add(q)
            retrieval = self.vector_retriever.retrieve(
                q,
                top_k=top_k,
                similarity_threshold=similarity_threshold,
            )
            query_traces.append(retrieval.stats)
            aggregated_docs.extend(self._tag_documents(retrieval.documents, q))

        # 2) 构造查询扩展
        expansion_queries, prf_queries = self._build_expansion_queries(
            question,
            queries,
            aggregated_docs,
            enable_prf,
            expansion_limit,
        )

        # 3) 执行扩展检索
        for q in expansion_queries + prf_queries:
            if q in seen_queries:
                continue
            seen_queries.add(q)
            retrieval = self.vector_retriever.retrieve(
                q,
                top_k=max(3, top_k // 2),
                similarity_threshold=max(similarity_threshold * 0.9, 0.2),
            )
            if retrieval.documents:
                query_traces.append(retrieval.stats)
                aggregated_docs.extend(self._tag_documents(retrieval.documents, q))

        # 4) 去重与过滤
        deduped_docs = self._deduplicate_documents(aggregated_docs)
        filtered_docs = self.content_filter.filter_documents(
            deduped_docs,
            question,
            max_docs=max(max_chunks * 3, 6),
        )
        if not filtered_docs:
            filtered_docs = deduped_docs[: max_chunks * 2]

        # 5) 重排序（可选）并做父/子分片均衡选取
        if use_reranking:
            ranked_docs = self.chunk_scorer.score(question, filtered_docs)
        else:
            ranked_docs = sorted(
                filtered_docs,
                key=lambda x: x.get("similarity", 0.0),
                reverse=True,
            )
        selected_chunks = self._select_balanced_chunks(ranked_docs, max_chunks)

        if not selected_chunks:
            response_time = time.time() - start_time
            fallback_answer = "抱歉，未能找到足够的文档来回答该问题。"
            logger.warning("统一RAG引擎：无相关分片，返回兜底答案")
            return EngineOutput(
                question=question,
                answer=fallback_answer,
                sources=[],
                retrieved_documents=deduped_docs,
                selected_chunks=[],
                query_traces=query_traces,
                expansion_log={
                    "initial_queries": list(queries),
                    "expansion_queries": expansion_queries,
                    "prf_queries": prf_queries,
                },
                response_time=response_time,
                confidence=0.15,
                fallback=True,
            )

        # 6) 构建回答
        llm_payload = self.llm_client.chat(
            question,
            context=selected_chunks,
            system_prompt=self.system_prompt,
        )
        answer_text = llm_payload.get("answer", "")

        response_time = time.time() - start_time
        sources = [self._format_source(doc) for doc in selected_chunks]
        confidence = self._estimate_confidence(selected_chunks)

        return EngineOutput(
            question=question,
            answer=answer_text,
            sources=sources,
            retrieved_documents=deduped_docs,
            selected_chunks=selected_chunks,
            query_traces=query_traces,
            expansion_log={
                "initial_queries": list(queries),
                "expansion_queries": expansion_queries,
                "prf_queries": prf_queries,
            },
            response_time=response_time,
            confidence=confidence,
            llm_metadata=llm_payload,
        )

    def _prepare_initial_queries(
        self, question: str, sub_questions: Optional[Sequence[str]]
    ) -> List[str]:
        prepared: List[str] = []
        if sub_questions:
            prepared.extend([q.strip() for q in sub_questions if q and q.strip()])
        if question.strip() not in prepared:
            prepared.insert(0, question.strip())
        return list(dict.fromkeys(prepared))

    def _build_expansion_queries(
        self,
        question: str,
        base_queries: Sequence[str],
        aggregated_docs: Sequence[Dict[str, Any]],
        enable_prf: bool,
        limit: int,
    ) -> Tuple[List[str], List[str]]:
        expansion_candidates: List[str] = []
        prf_candidates: List[str] = []

        try:
            for q in base_queries:
                expansion_candidates.extend(self.query_expander.expand_query(q))
                expansion_candidates.extend(
                    self.intelligent_expander.expand_query(q, list(aggregated_docs))
                )
        except Exception as exc:
            logger.warning(f"查询扩展失败，使用部分结果: {exc}")

        expansion_candidates = list(dict.fromkeys(expansion_candidates))
        expansion_candidates = [q for q in expansion_candidates if q.strip()]

        if enable_prf and self.prf_expander:
            try:
                prf_candidates = self.prf_expander.expand(
                    question,
                    top_docs=list(aggregated_docs),
                    max_expansions=3,
                )
            except Exception as exc:
                logger.warning(f"PRF 扩展失败: {exc}")
                prf_candidates = []

        if limit > 0:
            expansion_candidates = expansion_candidates[:limit]

        return expansion_candidates, prf_candidates

    def _tag_documents(
        self, documents: Sequence[Dict[str, Any]], query: str
    ) -> List[Dict[str, Any]]:
        tagged: List[Dict[str, Any]] = []
        for doc in documents:
            doc_copy = dict(doc)
            doc_copy.setdefault("chunk_id", doc_copy.get("parent_id"))
            doc_copy.setdefault("similarity", 0.0)
            existing = doc_copy.get("source_queries") or []
            if isinstance(existing, list):
                merged = set(existing)
            else:
                merged = {query}
            merged.add(query)
            doc_copy["source_queries"] = list(merged)
            tagged.append(doc_copy)
        return tagged

    def _deduplicate_documents(
        self, documents: Sequence[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        dedup: Dict[str, Dict[str, Any]] = {}
        for doc in documents:
            key = doc.get("chunk_id") or doc.get("parent_id")
            if not key:
                key = f"anon_{len(dedup)}"
            current = dedup.get(key)
            if current is None or doc.get("similarity", 0.0) > current.get(
                "similarity", 0.0
            ):
                new_doc = dict(doc)
                new_doc["chunk_id"] = key
                dedup[key] = new_doc
            else:
                # 合并来源查询
                current_sources = set(current.get("source_queries", []))
                current_sources.update(doc.get("source_queries", []))
                current["source_queries"] = list(current_sources)
        return sorted(
            dedup.values(), key=lambda x: x.get("similarity", 0.0), reverse=True
        )

    @staticmethod
    def _is_parent_chunk(doc: Dict[str, Any]) -> bool:
        chunk_id = str(doc.get("chunk_id", ""))
        return chunk_id.startswith("parent_") or bool(doc.get("is_parent"))

    def _select_balanced_chunks(
        self, ranked_docs: Sequence[Dict[str, Any]], max_chunks: int
    ) -> List[Dict[str, Any]]:
        """优先保留至少 1 个父分片，再补充子分片，避免上下文丢失。"""
        if max_chunks <= 0 or not ranked_docs:
            return []

        parents = [d for d in ranked_docs if self._is_parent_chunk(d)]
        children = [d for d in ranked_docs if not self._is_parent_chunk(d)]

        selected: List[Dict[str, Any]] = []
        if parents and max_chunks > 1:
            selected.append(parents[0])

        remaining_slots = max_chunks - len(selected)
        selected.extend(children[:remaining_slots])

        if len(selected) < max_chunks:
            selected_ids = {d.get("chunk_id") for d in selected}
            for doc in ranked_docs:
                if doc.get("chunk_id") in selected_ids:
                    continue
                selected.append(doc)
                selected_ids.add(doc.get("chunk_id"))
                if len(selected) >= max_chunks:
                    break

        return selected

    def _format_source(self, doc: Dict[str, Any]) -> Dict[str, Any]:
        content = doc.get("content", "")
        preview = content[:200] + ("..." if len(content) > 200 else "")
        return {
            "chunk_id": doc.get("chunk_id"),
            "filename": doc.get("filename"),
            "similarity": round(float(doc.get("similarity", 0.0)), 4),
            "content_preview": preview,
            "metadata": doc.get("metadata", {}),
            "source_queries": doc.get("source_queries", []),
        }

    def _estimate_confidence(self, docs: Sequence[Dict[str, Any]]) -> float:
        if not docs:
            return 0.1

        llm_scores = [d.get("llm_score") for d in docs if d.get("llm_score") is not None]
        sim_scores = [d.get("similarity") for d in docs if d.get("similarity") is not None]

        llm_avg = sum(llm_scores) / len(llm_scores) if llm_scores else 0.55
        sim_avg = sum(sim_scores) / len(sim_scores) if sim_scores else 0.5

        confidence = 0.6 * llm_avg + 0.4 * sim_avg
        return max(0.1, min(0.95, confidence))

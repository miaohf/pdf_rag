#!/usr/bin/env python3
"""
智能内容过滤器

基于语义相似度的文档内容过滤，替代硬编码关键词映射。
"""

import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

from src.utils.config import Config
from src.utils.logger import get_logger
from src.core.embeddings import EmbeddingModel

logger = get_logger(__name__)


@dataclass
class FilterResult:
    """过滤结果"""
    keep: bool
    relevance_score: float
    reason: str
    matched_concepts: List[str]


class SemanticContentFilter:
    """基于语义相似度的内容过滤器"""
    
    def __init__(self, config: Config, embedding_model: Optional[EmbeddingModel] = None):
        """
        初始化语义内容过滤器
        
        Args:
            config: 配置对象
            embedding_model: 嵌入模型
        """
        self.config = config
        self.embedding_model = embedding_model or EmbeddingModel(config)
        
        # 语义相似度阈值
        self.semantic_threshold = 0.6
        self.keyword_bonus = 0.2  # 关键词匹配额外加分
        
        logger.info("语义内容过滤器初始化完成")
    
    def filter_documents(
        self, 
        docs: List[Dict[str, Any]], 
        question: str,
        min_relevance: float = 0.5,
        max_docs: int = 5
    ) -> List[Dict[str, Any]]:
        """
        过滤文档，保留最相关的内容
        
        Args:
            docs: 文档列表
            question: 用户问题
            min_relevance: 最小相关度阈值
            max_docs: 最大文档数量
            
        Returns:
            过滤后的文档列表
        """
        if not docs:
            return docs
        
        logger.info(f"开始语义过滤 {len(docs)} 个文档，问题: {question}")
        
        # 计算每个文档的相关性评分
        scored_docs = []
        
        for doc in docs:
            filter_result = self._evaluate_document_relevance(doc, question)
            
            if filter_result.keep:
                doc_with_score = doc.copy()
                doc_with_score['filter_score'] = filter_result.relevance_score
                doc_with_score['filter_reason'] = filter_result.reason
                doc_with_score['matched_concepts'] = filter_result.matched_concepts
                scored_docs.append(doc_with_score)
                
                logger.debug(f"保留文档 {doc.get('chunk_id', 'unknown')[:8]}... "
                           f"(评分: {filter_result.relevance_score:.3f}, 原因: {filter_result.reason})")
            else:
                logger.debug(f"过滤文档 {doc.get('chunk_id', 'unknown')[:8]}... "
                           f"(评分: {filter_result.relevance_score:.3f}, 原因: {filter_result.reason})")
        
        # 按评分排序并限制数量
        scored_docs.sort(key=lambda x: x['filter_score'], reverse=True)
        final_docs = scored_docs[:max_docs]
        
        # 如果过滤后文档太少，保留一些高质量的原始文档
        if len(final_docs) < 2 and len(docs) >= 2:
            logger.warning(f"过滤后文档过少({len(final_docs)})，保留前2个原始文档")
            for doc in docs[:2]:
                if doc not in final_docs:
                    doc['filter_score'] = 0.3  # 给予较低分数
                    doc['filter_reason'] = '保底文档'
                    final_docs.append(doc)
        
        logger.info(f"语义过滤完成: {len(docs)} -> {len(final_docs)} 个文档")
        return final_docs
    
    def _evaluate_document_relevance(
        self, 
        doc: Dict[str, Any], 
        question: str
    ) -> FilterResult:
        """
        评估文档与问题的相关性
        
        Args:
            doc: 文档对象
            question: 用户问题
            
        Returns:
            过滤结果
        """
        content = doc.get('content', '')
        
        # 1. 语义相似度评分 (主要指标)
        semantic_score = self._calculate_semantic_similarity(content, question)
        
        # 2. 关键词匹配评分 (辅助指标)
        keyword_score, matched_keywords = self._calculate_keyword_relevance(content, question)
        
        # 3. 内容质量评分 (长度、结构等)
        quality_score = self._calculate_content_quality(content)
        
        # 综合评分 (加权平均)
        relevance_score = (
            semantic_score * 0.6 +      # 语义相似度权重最高
            keyword_score * 0.3 +       # 关键词匹配
            quality_score * 0.1         # 内容质量
        )
        
        # 判断是否保留
        keep = relevance_score >= self.semantic_threshold
        
        # 生成解释
        if keep:
            reason = f"语义匹配({semantic_score:.2f}) + 关键词({keyword_score:.2f})"
        else:
            reason = f"相关性不足({relevance_score:.2f} < {self.semantic_threshold})"
        
        return FilterResult(
            keep=keep,
            relevance_score=relevance_score,
            reason=reason,
            matched_concepts=matched_keywords
        )
    
    def _calculate_semantic_similarity(self, content: str, question: str) -> float:
        """计算语义相似度"""
        try:
            # 生成文档和问题的嵌入向量
            doc_embedding = self.embedding_model.embed_text(content[:500])  # 限制长度
            question_embedding = self.embedding_model.embed_text(question)
            
            if not doc_embedding or not question_embedding:
                return 0.0
            
            # 计算余弦相似度
            import numpy as np
            
            doc_vec = np.array(doc_embedding)
            question_vec = np.array(question_embedding)
            
            # 余弦相似度
            dot_product = np.dot(doc_vec, question_vec)
            norm_doc = np.linalg.norm(doc_vec)
            norm_question = np.linalg.norm(question_vec)
            
            if norm_doc == 0 or norm_question == 0:
                return 0.0
            
            similarity = dot_product / (norm_doc * norm_question)
            return max(0.0, float(similarity))  # 确保非负
            
        except Exception as e:
            logger.warning(f"计算语义相似度失败: {e}")
            return 0.0
    
    def _calculate_keyword_relevance(self, content: str, question: str) -> Tuple[float, List[str]]:
        """计算关键词相关性"""
        content_lower = content.lower()
        question_lower = question.lower()
        
        # 提取问题中的重要词汇 (去除停用词)
        stop_words = {'的', '是', '在', '有', '和', '与', '或', '但', '如果', '那么', '这', '那', '什么', '怎么', '为什么', '多少',
                     'the', 'is', 'in', 'and', 'or', 'but', 'if', 'then', 'this', 'that', 'what', 'how', 'why', 'how much'}
        
        question_words = [word for word in re.findall(r'\b\w+\b', question_lower) 
                         if len(word) > 1 and word not in stop_words]
        
        # 计算关键词匹配
        matched_keywords = []
        total_words = len(question_words)
        
        if total_words == 0:
            return 0.0, []
        
        for word in question_words:
            if word in content_lower:
                matched_keywords.append(word)
        
        # 关键词匹配率
        match_ratio = len(matched_keywords) / total_words
        
        # 动态词汇加权：基于词频和长度
        weighted_score = 0.0
        total_weight = 0.0
        
        for word in question_words:
            # 基于词长度的权重：更长的词通常更重要
            length_weight = min(2.0, len(word) / 4.0)
            
            # 基于在内容中的稀有度：出现次数少的词更重要
            word_count_in_content = content_lower.count(word)
            rarity_weight = 1.0 + (1.0 / max(1, word_count_in_content))
            
            # 综合权重
            weight = length_weight * rarity_weight
            total_weight += weight
            
            if word in matched_keywords:
                weighted_score += weight
        
        if total_weight > 0:
            final_score = weighted_score / total_weight
        else:
            final_score = match_ratio
        
        return min(1.0, final_score), matched_keywords
    
    def _calculate_content_quality(self, content: str) -> float:
        """计算内容质量评分"""
        if not content:
            return 0.0
        
        quality_score = 0.5  # 基础分数
        
        # 长度评分 (适中的长度更好)
        length = len(content)
        if 100 <= length <= 500:
            quality_score += 0.3
        elif 50 <= length < 100 or 500 < length <= 1000:
            quality_score += 0.2
        elif length < 50:
            quality_score += 0.1
        
        # 结构评分 (包含数字、标点等)
        if re.search(r'\d+', content):  # 包含数字
            quality_score += 0.1
        
        if re.search(r'[.。!！?？]', content):  # 包含句子结构
            quality_score += 0.1
        
        return min(1.0, quality_score)


class HybridContentFilter:
    """混合内容过滤器：结合语义分析和统计过滤"""
    
    def __init__(self, config: Config, embedding_model: Optional[EmbeddingModel] = None):
        """初始化混合过滤器"""
        self.semantic_filter = SemanticContentFilter(config, embedding_model)
        self.config = config
        
        # 完全移除硬编码规则，改用统计和语义方法
    
    def filter_documents(
        self, 
        docs: List[Dict[str, Any]], 
        question: str,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        使用混合策略过滤文档
        
        Args:
            docs: 文档列表
            question: 用户问题
            **kwargs: 其他参数
            
        Returns:
            过滤后的文档列表
        """
        try:
            # 优先使用语义过滤
            filtered_docs = self.semantic_filter.filter_documents(docs, question, **kwargs)
            
            # 如果语义过滤结果太少，使用降低阈值的语义过滤作为补充
            if len(filtered_docs) < 2:
                logger.info("语义过滤结果较少，启用降阈值补充")
                relaxed_filtered = self._relaxed_semantic_filter(docs, question, kwargs.get('max_docs', 5))
                
                # 合并结果，去重
                seen_ids = {doc.get('chunk_id') for doc in filtered_docs}
                for doc in relaxed_filtered:
                    if doc.get('chunk_id') not in seen_ids:
                        doc['filter_score'] = doc.get('filter_score', 0.3)  # 保持原始评分
                        doc['filter_reason'] = '降阈值补充'
                        filtered_docs.append(doc)
                        if len(filtered_docs) >= kwargs.get('max_docs', 5):
                            break
            
            return filtered_docs
            
        except Exception as e:
            logger.error(f"混合过滤失败，使用基础文档返回: {e}")
            return self._basic_fallback_filter(docs, question)
    
    def _relaxed_semantic_filter(self, docs: List[Dict[str, Any]], question: str, max_docs: int = 5) -> List[Dict[str, Any]]:
        """降低阈值的语义过滤（补充方案）"""
        logger.info("使用降阈值语义过滤作为补充")
        
        # 临时降低语义阈值
        original_threshold = self.semantic_filter.semantic_threshold
        self.semantic_filter.semantic_threshold = 0.3  # 降低到0.3
        
        try:
            # 重新过滤，使用更宽松的标准
            relaxed_docs = self.semantic_filter.filter_documents(
                docs, 
                question, 
                min_relevance=0.3, 
                max_docs=max_docs
            )
            return relaxed_docs
        finally:
            # 恢复原始阈值
            self.semantic_filter.semantic_threshold = original_threshold
    
    def _basic_fallback_filter(self, docs: List[Dict[str, Any]], question: str) -> List[Dict[str, Any]]:
        """基础回退过滤（最后方案）"""
        if not docs:
            return []
        
        logger.warning("使用基础回退过滤策略")
        
        # 基于文档长度和基本文本匹配的简单过滤
        scored_docs = []
        question_words = set(question.lower().split())
        
        for doc in docs:
            content = doc.get('content', '').lower()
            content_words = set(content.split())
            
            # 计算词汇重叠度
            if question_words and content_words:
                overlap = len(question_words & content_words) / len(question_words)
            else:
                overlap = 0.0
            
            # 基于长度的质量评分
            length_score = min(1.0, len(content) / 200) * 0.5
            
            total_score = overlap * 0.7 + length_score * 0.3
            
            if total_score > 0.1:  # 非常低的阈值
                doc_copy = doc.copy()
                doc_copy['filter_score'] = total_score
                doc_copy['filter_reason'] = f"基础匹配(重叠度:{overlap:.2f})"
                scored_docs.append(doc_copy)
        
        # 排序并返回前几个
        scored_docs.sort(key=lambda x: x['filter_score'], reverse=True)
        return scored_docs[:5] 
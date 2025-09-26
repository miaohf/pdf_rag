"""
RAG服务层

提供高级的问答接口和服务管理。
"""

import time
import uuid
from typing import Dict, Any, Optional, AsyncGenerator, Generator, List

from src.utils.config import Config
from src.utils.logger import get_logger
from src.core.database import Database
from src.rag.prf_expander import LLMPRFExpander
from src.rag.chunk_scorer import ChunkScorer

logger = get_logger(__name__)


class RAGService:
    """RAG服务"""
    
    def __init__(self, config: Config, database: Optional[Database] = None):
        """
        初始化RAG服务
        
        Args:
            config: 配置对象
            database: 数据库对象
        """
        self.config = config
        self.db = database or Database(config)
        
        # 初始化核心组件
        from src.core.embeddings import EmbeddingModel
        from src.core.llm import LLMClient
        from src.core.vector_store import VectorStore
        from src.core.hierarchical_retriever import HierarchicalRetriever
        from src.rag.agentic_engine import AgenticRAGEngine
        from src.rag.tools import ToolManager
        from src.rag.conversation import ConversationManager, MultiTurnRagProcessor
        from src.rag.content_filter import HybridContentFilter
        
        try:
            self.embedding_model = EmbeddingModel(config)
            self.llm_client = LLMClient(config)
            self.vector_store = VectorStore(config, self.db, self.embedding_model)
            self.hierarchical_retriever = HierarchicalRetriever(config, self.db, self.vector_store)
            self.agentic_engine = AgenticRAGEngine(config, self.llm_client)
            self.tool_manager = ToolManager(config, self.vector_store)
            self.conversation_manager = ConversationManager(config, self.db)
            self.content_filter = HybridContentFilter(config)
            
            # 通用 PRF 查询扩展器（无硬编码）
            self.prf_expander = LLMPRFExpander(config, self.llm_client)
            
            # LLM 分片相关性打分器（通用、无硬编码）
            self.chunk_scorer = ChunkScorer(config, self.llm_client)
            
            logger.info("RAG服务初始化完成")
            
        except Exception as e:
            logger.error(f"RAG服务初始化失败: {e}")
            raise
    
    async def query(self, question: str, top_k: int = 5, max_chunks: int = 3, 
                   use_reranking: bool = True, 
                   similarity_threshold: float = 0.3, **kwargs) -> Dict[str, Any]:
        """查询处理"""
        start_time = time.time()
        
        # 参数验证和默认值处理
        top_k = top_k or 5
        max_chunks = max_chunks or 3
        similarity_threshold = similarity_threshold or 0.3
        
        logger.info(f"收到查询: {question}")
        logger.info(f"查询参数: top_k={top_k}, max_chunks={max_chunks}, similarity_threshold={similarity_threshold}")
        
        # 1) 首先使用大模型分解问题
        logger.info("开始问题分解...")
        plan = self.agentic_engine.planner.plan_query(question)
        sub_questions = plan.sub_questions or [question]
        logger.info(f"问题分解完成，得到 {len(sub_questions)} 个子问题: {sub_questions}")
        
        # 2) 针对每个子问题独立进行相似度查询
        all_retrieved_chunks: List[Dict[str, Any]] = []
        sub_answers: List[Dict[str, Any]] = []
        
        for i, sub_q in enumerate(sub_questions, 1):
            logger.info(f"处理子问题 {i}/{len(sub_questions)}: {sub_q}")
            
            # 对子问题进行检索
            hr = self.hierarchical_retriever.hierarchical_search(
                query=sub_q,
                top_k=top_k,
                similarity_threshold=similarity_threshold
            )
            
            # 收集检索到的分片
            sub_chunks = hr.child_chunks + hr.parent_contexts
            all_retrieved_chunks.extend(sub_chunks)
            
            # 使用LLM对检索到的分片进行相关性打分
            scored_chunks = self.chunk_scorer.score(sub_q, sub_chunks)
            
            # 为每个子问题选择最相关的分片
            # 确保父分片和子分片都有机会被选中
            parent_chunks = [c for c in scored_chunks if str(c.get("chunk_id", "")).startswith("parent_")]
            child_chunks = [c for c in scored_chunks if not str(c.get("chunk_id", "")).startswith("parent_")]
            
            # 每个子问题选择：至少1个父分片（如果有）+ 2个子分片
            selected_for_sub = []
            if parent_chunks:
                selected_for_sub.extend(parent_chunks[:1])  # 选择1个最佳父分片
            selected_for_sub.extend(child_chunks[:2])  # 选择2个最佳子分片
            
            # 如果没有足够的分片，从剩余的scored中补充
            if len(selected_for_sub) < 3:
                remaining = [c for c in scored_chunks if c not in selected_for_sub]
                selected_for_sub.extend(remaining[:3-len(selected_for_sub)])
            
            logger.info(f"子问题 {i} 选择了 {len(selected_for_sub)} 个分片: {[c.get('chunk_id') for c in selected_for_sub]}")
            
            # 生成子问题的答案
            sub_context = self._build_context(selected_for_sub)
            sub_prompt = self._build_prompt(sub_q, sub_context)
            try:
                sub_answer_text = self.llm_client.generate(sub_prompt, temperature=0.1)
                logger.info(f"子问题 {i} 答案生成完成")
            except Exception as e:
                logger.warning(f"子问题 {i} 回答失败: {e}")
                sub_answer_text = ""
            
            sub_answers.append({
                "sub_question": sub_q,
                "answer": sub_answer_text,
                "chunks": [c.get("chunk_id") for c in selected_for_sub if c.get("chunk_id")]
            })
        
        # 3) 去重所有检索到的分片
        seen_chunk_ids = set()
        unique_chunks: List[Dict[str, Any]] = []
        for chunk in all_retrieved_chunks:
            cid = chunk.get("chunk_id")
            if cid and cid not in seen_chunk_ids:
                seen_chunk_ids.add(cid)
                unique_chunks.append(chunk)
        
        # 按相似度排序
        unique_chunks.sort(key=lambda x: x.get("similarity", 0.0), reverse=True)
        logger.info(f"合并后唯一文档数: {len(unique_chunks)}")
        logger.info(f"chunk_ids: {[r.get('chunk_id') for r in unique_chunks[:10]]}")  # 只显示前10个
        
        if not unique_chunks:
            logger.warning("未找到相关文档")
            return {
                "answer": "抱歉，我没有找到相关的文档来回答这个问题。",
                "confidence": 0.0,
                "sources": [],
                "response_time": time.time() - start_time,
                "query_id": str(uuid.uuid4()),
                "hierarchical_info": {}
            }
        
        # 4) 基于子问题答案生成最终综合答案
        if sub_answers and len(sub_answers) > 1:
            # 如果有多个子问题，汇总子答案生成最终答案
            sub_answers_text = ""
            for i, sub_answer in enumerate(sub_answers, 1):
                sub_answers_text += f"\n\n**子问题{i}**: {sub_answer['sub_question']}\n"
                sub_answers_text += f"**答案{i}**: {sub_answer['answer']}"
            
            final_prompt = f"""
你是一个专业的法律法规智能助手。请基于以下子问题的答案，为用户提供一个完整、准确的综合回答。

原始问题：{question}

子问题及其答案：{sub_answers_text}

请将上述子问题的答案整合为一个完整、连贯的回答，要求：
1. 保持所有子答案的准确性，不要编造信息
2. 回答要结构清晰、易于理解
3. 如果子答案中有引用文档，请保留这些引用
4. 保持专业、客观的语调

综合回答：
"""
            logger.info("基于子问题答案生成最终综合答案...")
        else:
            # 如果只有一个子问题，直接使用其答案
            if sub_answers:
                logger.info("只有一个子问题，直接使用其答案")
                final_answer = sub_answers[0]['answer']
                sources = [self._format_source(doc) for doc in unique_chunks[:max_chunks]]
                return {
                    "answer": final_answer,
                    "confidence": 0.8,
                    "sources": sources,
                    "response_time": time.time() - start_time,
                    "query_id": str(uuid.uuid4()),
                    "hierarchical_info": {
                        "sub_questions": sub_questions,
                        "sub_answers": sub_answers,
                        "total_chunks_retrieved": len(unique_chunks),
                        "final_chunks_used": len(unique_chunks[:max_chunks])
                    }
                }
            else:
                # 备用方案：基于分片生成答案
                logger.info("无子答案，使用备用方案基于分片生成答案")
                final_chunks = unique_chunks[:max_chunks]
                final_context = self._build_context(final_chunks)
                final_prompt = self._build_prompt(question, final_context)
        
        try:
            final_answer = self.llm_client.generate(final_prompt, temperature=0.1)
            logger.info("最终答案生成完成")
            
            sources = [self._format_source(doc) for doc in unique_chunks[:max_chunks]]
            return {
                "answer": final_answer,
                "confidence": 0.8,
                "sources": sources,
                "response_time": time.time() - start_time,
                "query_id": str(uuid.uuid4()),
                "hierarchical_info": {
                    "sub_questions": sub_questions,
                    "sub_answers": sub_answers,
                    "total_chunks_retrieved": len(unique_chunks),
                    "final_chunks_used": len(unique_chunks[:max_chunks])
                }
            }
        except Exception as e:
            logger.error(f"生成答案失败: {e}")
            return {
                "answer": f"抱歉，生成答案时出现错误: {str(e)}",
                "confidence": 0.0,
                "sources": [],
                "response_time": time.time() - start_time,
                "query_id": str(uuid.uuid4()),
                "hierarchical_info": {}
            }
    
    def query_stream(self, question: str, **kwargs) -> Generator[str, None, None]:
        """
        流式查询
        
        Args:
            question: 用户问题
            **kwargs: 其他参数
            
        Yields:
            响应流
        """
        logger.info(f"收到流式查询: {question}")
        
        # TODO: 实现流式RAG
        # 临时返回模拟流式响应
        response = f'这是对问题"{question}"的流式模拟回答。RAG功能正在开发中，请等待完整实现。'
        
        for word in response.split():
            time.sleep(0.1)  # 模拟流式延迟
            yield word + ' '
    
    async def query_async(self, question: str, **kwargs) -> Dict[str, Any]:
        """
        异步查询
        
        Args:
            question: 用户问题
            **kwargs: 其他参数
            
        Returns:
            查询结果
        """
        # 简单的异步包装
        return await self.query(question, **kwargs)
    
    async def query_stream_async(self, question: str, **kwargs) -> AsyncGenerator[str, None]:
        """
        异步流式查询
        
        Args:
            question: 用户问题
            **kwargs: 其他参数
            
        Yields:
            响应流
        """
        logger.info(f"收到异步流式查询: {question}")
        
        # TODO: 实现异步流式RAG
        # 临时返回模拟流式响应
        response = f'这是对问题"{question}"的异步流式模拟回答。RAG功能正在开发中，请等待完整实现。'
        
        for word in response.split():
            import asyncio
            await asyncio.sleep(0.1)  # 模拟异步延迟
            yield word + ' '
    
    def _save_query_history(self, question: str, result: Dict[str, Any]):
        """
        保存查询历史（使用ORM）
        
        Args:
            question: 问题
            result: 查询结果
        """
        try:
            from src.core.models import QueryHistory
            import json
            
            with self.db.get_session() as session:
                query_id = result.get('query_id', str(uuid.uuid4()))
                
                # 使用ORM创建查询历史记录
                query_history = QueryHistory(
                    id=query_id,
                    session_id=result.get('session_id'),
                    user_id=result.get('user_id'),
                    query_text=question,
                    query_type=result.get('query_type', 'standard'),
                    response_text=result.get('answer', ''),
                    confidence=str(result.get('confidence', 0)),
                    sources_count=len(result.get('sources', [])),
                    response_time=str(result.get('response_time', 0)),
                    query_metadata={
                        'sources': result.get('sources', []),
                        'has_tool_calls': result.get('has_tool_calls', False),
                        'agentic_type': result.get('agentic_type'),
                        'reasoning_steps': result.get('reasoning_steps', [])
                    }
                )
                
                session.add(query_history)
                logger.info(f"保存查询历史: {query_id}")
                
        except Exception as e:
            logger.error(f"保存查询历史失败: {e}")
    
    def _check_rcp_completeness(self, retrieved_docs: List[Dict[str, Any]], question: str) -> Dict[str, Any]:
        """
        检查RCP系统信息完整性
        
        Args:
            retrieved_docs: 检索到的文档
            question: 用户问题
            
        Returns:
            完整性检查结果
        """
        if "RCP" not in question and "Remote Control Parking" not in question:
            return {"needs_completion": False, "missing_info": []}
        
        # RCP系统关键参数检查
        rcp_parameters = {
            "法规依据": ["legislation", "regulation", "technical guideline", "cap 374a", "40c"],
            "最大行驶距离": ["12 metres", "12 meters", "travel distance", "vehicle travel", "12m"],
            "速度限制": ["2 km/h", "2km/h", "maximum speed", "vehicle speed", "2 km"],
            "最大操作距离": ["6 metres", "6 meters", "operation distance", "control distance", "handheld distance", "6m"]
        }
        
        missing_info = []
        found_info = {}
        
        # 检查每个参数是否被覆盖
        for param_name, keywords in rcp_parameters.items():
            param_found = False
            for doc in retrieved_docs:
                content = doc.get("content", "").lower()
                if any(keyword.lower() in content for keyword in keywords):
                    param_found = True
                    found_info[param_name] = doc
                    break
            
            if not param_found:
                missing_info.append(param_name)
        
        return {
            "needs_completion": len(missing_info) > 0,
            "missing_info": missing_info,
            "found_info": found_info
        }
    
    def _enhanced_search(self, question: str, top_k: int, similarity_threshold: float) -> Dict[str, Any]:
        """
        增强查询搜索
        
        Args:
            question: 原始问题
            top_k: 返回数量
            similarity_threshold: 相似度阈值
            
        Returns:
            增强搜索结果
        """
        # 生成增强查询
        enhanced_queries = self._generate_enhanced_queries(question)
        logger.info(f"生成 {len(enhanced_queries)} 个增强查询")
        
        # 收集所有结果，去重并按相似度排序
        all_results = {}  # chunk_id -> best_result
        original_count = 0
        
        for i, query in enumerate(enhanced_queries):
            try:
                results = self.vector_store.search_similar(
                    query=query,
                    top_k=top_k,
                    similarity_threshold=similarity_threshold
                )
                
                if i == 0:  # 记录原始查询的结果数量
                    original_count = len(results)
                
                for result in results:
                    chunk_id = result['chunk_id']
                    similarity = result['similarity']
                    
                    # 保留每个分片的最高相似度结果
                    if chunk_id not in all_results or similarity > all_results[chunk_id]['similarity']:
                        all_results[chunk_id] = result
                        
            except Exception as e:
                logger.warning(f"增强查询失败: {query} - {e}")
        
        # 按相似度排序并限制数量
        sorted_results = sorted(all_results.values(), key=lambda x: x['similarity'], reverse=True)
        final_results = sorted_results[:top_k]
        
        return {
            'documents': final_results,
            'original_count': original_count,
            'enhanced_count': len(all_results),
            'final_count': len(final_results)
        }
    
    def _generate_enhanced_queries(self, question: str) -> list:
        """
        生成增强查询
        
        Args:
            question: 原始问题
            
        Returns:
            增强查询列表
        """
        enhanced_queries = [question]  # 始终包含原始查询
        
        # 中英文映射
        english_mappings = {
            "Remote Control Parking": ["Remote Control Parking", "RCP"],
            "最大速度": ["maximum speed", "speed limit", "maximum vehicle speed"],
            "速度限制": ["speed limit", "maximum speed", "vehicle speed", "speed restriction"],
            "限制": ["limit", "restriction", "exceeding", "not exceeding"],
            "是多少": ["how much", "what is", "value", "amount"],
            "技术要求": ["technical requirements", "specifications", "requirements"],
            "规定": ["regulations", "rules", "requirements"],
            "系统": ["system", "device"],
            "操作": ["operation", "control", "use"],

        }
        
        # 提取英文关键词
        english_keywords = []
        for chinese_term, english_terms in english_mappings.items():
            if chinese_term in question:
                english_keywords.extend(english_terms)
        
        # 生成英文查询组合
        if english_keywords:
            # 基础英文查询
            enhanced_queries.append(" ".join(english_keywords[:3]))
            
            # 针对具体问题的优化查询
            if "Remote Control Parking" in question and "速度" in question:
                enhanced_queries.extend([
                    "Maximum vehicle speed during RCP operation",
                    "RCP speed limit",
                    "vehicle speed RCP operation",
                    "RCP maximum speed limit",
                    "Remote Control Parking speed restriction"
                ])
            
            # 新增：针对RCP系统操作距离的专门查询
            if "Remote Control Parking" in question or "RCP" in question:
                enhanced_queries.extend([
                    "RCP operation distance",
                    "RCP control distance",
                    "RCP handheld device distance",
                    "RCP maximum control range",
                    "RCP remote operation distance",
                    "RCP device communication distance",
                    "RCP handheld control range"
                ])
        
        # 添加更多上下文查询
        if "RCP" in question or "Remote Control Parking" in question:
            enhanced_queries.extend([
                "RCP technical requirements",
                "Remote Control Parking specifications",
                "RCP system parameters",
                "RCP device specifications",
                "RCP operation parameters"
            ])
        
        # 去重并返回
        return list(dict.fromkeys(enhanced_queries))  # 保持顺序的去重
    
    def _filter_relevant_docs(self, docs: List[Dict[str, Any]], question: str) -> List[Dict[str, Any]]:
        """
        预过滤文档，只保留真正包含相关信息的分片
        
        Args:
            docs: 检索到的文档列表
            question: 用户问题
            
        Returns:
            过滤后的文档列表
        """
        if not docs:
            return docs
        
        # 定义关键词过滤规则
        keyword_filters = {
            "速度限制": ["speed", "km/h", "limit", "maximum", "exceeding", "vehicle speed"],
            "技术要求": ["technical", "requirements", "specifications", "comply"],
            "申请流程": ["application", "documents", "certificate", "approval"],
            "系统功能": ["system", "function", "operation", "control", "device"]
        }
        
        # 提取问题关键词
        question_keywords = []
        if "速度" in question and "限制" in question:
            question_keywords = keyword_filters["速度限制"]
        elif "技术" in question and "要求" in question:
            question_keywords = keyword_filters["技术要求"]
        elif "申请" in question:
            question_keywords = keyword_filters["申请流程"]
        else:
            # 默认保留所有文档
            return docs
        
        filtered_docs = []
        
        for doc in docs:
            content = doc.get('content', '').lower()
            
            # 检查是否包含相关关键词
            keyword_matches = sum(1 for keyword in question_keywords if keyword.lower() in content)
            
            # 对于速度限制问题，进行更严格的过滤
            if "速度" in question and "限制" in question:
                # 必须包含速度相关的具体数值或描述
                speed_indicators = ["km/h", "speed", "maximum", "limit", "exceeding", "2 km", "2km"]
                has_speed_info = any(indicator in content for indicator in speed_indicators)
                
                if has_speed_info and keyword_matches >= 2:
                    filtered_docs.append(doc)
                    logger.debug(f"保留分片 {doc.get('chunk_id', 'unknown')[:8]}... - 包含速度信息")
                else:
                    logger.debug(f"过滤分片 {doc.get('chunk_id', 'unknown')[:8]}... - 不包含相关速度信息")
            else:
                # 其他问题使用通用关键词匹配
                if keyword_matches >= 1:
                    filtered_docs.append(doc)
        
        # 如果过滤后没有文档，保留原始结果避免无法回答
        if not filtered_docs:
            logger.warning("过滤后无相关文档，保留原始结果")
            return docs[:3]  # 至少保留前3个
        
        # 保持相似度排序并限制数量
        filtered_docs.sort(key=lambda x: x.get('similarity', 0), reverse=True)
        return filtered_docs[:5]  # 最多保留5个最相关的
    
    def _mock_response(self, question: str, start_time: float, error_msg: str = None) -> Dict[str, Any]:
        """
        创建模拟响应
        
        Args:
            question: 问题
            start_time: 开始时间
            error_msg: 错误信息
            
        Returns:
            模拟响应
        """
        end_time = time.time()
        
        if error_msg:
            answer = f"抱歉，{error_msg}"
        else:
            answer = f'这是对问题"{question}"的模拟回答。RAG功能正在开发中或服务不可用。'
        
        result = {
            'answer': answer,
            'confidence': 0.1,
            'sources': [],
            'query_id': str(uuid.uuid4()),
            'response_time': end_time - start_time,
            'model': 'mock',
            'retrieved_docs': 0
        }
        
        self._save_query_history(question, result)
        return result
    
    def _rerank_documents(self, documents: List[Dict[str, Any]], question: str) -> List[Dict[str, Any]]:
        """
        重排序文档（简单实现）
        
        Args:
            documents: 文档列表
            question: 查询问题
            
        Returns:
            重排序后的文档列表
        """
        # 简单的重排序：基于相似度和内容长度
        def score_doc(doc):
            similarity = doc.get('similarity', 0)
            content_length = len(doc.get('content', ''))
            
            # 计算综合分数
            length_score = min(content_length / 1000, 1.0)  # 长度分数
            final_score = similarity * 0.8 + length_score * 0.2
            
            return final_score
        
        return sorted(documents, key=score_doc, reverse=True)
    
    def _calculate_confidence(self, documents: List[Dict[str, Any]]) -> float:
        """
        计算回答置信度
        
        Args:
            documents: 检索到的文档列表
            
        Returns:
            置信度分数 (0-1)
        """
        if not documents:
            return 0.1
        
        # 基于文档数量和平均相似度计算置信度
        avg_similarity = sum(doc.get('similarity', 0) for doc in documents) / len(documents)
        doc_count_factor = min(len(documents) / 5, 1.0)  # 文档数量因子
        
        confidence = avg_similarity * 0.7 + doc_count_factor * 0.3
        return max(0.1, min(0.95, confidence))
    
    def _format_source(self, doc: Dict[str, Any]) -> Dict[str, Any]:
        """
        格式化文档来源信息
        
        Args:
            doc: 文档字典
            
        Returns:
            格式化的来源信息
        """
        return {
            'chunk_id': doc.get('chunk_id'),
            'filename': doc.get('filename'),
            'document_name': doc.get('document_name'),
            'similarity': round(doc.get('similarity', 0), 3),
            'content': doc.get('content', ''),  # 完整内容
            'content_preview': doc.get('content', '')[:200] + '...' if len(doc.get('content', '')) > 200 else doc.get('content', ''),
            'content_length': len(doc.get('content', '')),
            'metadata': doc.get('metadata', {}),
            
            # 前端定位信息
            'reference_format': f"【文档：{doc.get('filename', 'unknown')}，片段ID：{doc.get('chunk_id', 'unknown')}，相似度：{doc.get('similarity', 0):.3f}】",
            
            # 为前端提供完整的分片信息
            'chunk_info': {
                'id': doc.get('chunk_id'),
                'content': doc.get('content', ''),  # 确保这是完整的处理后内容
                'preview': doc.get('content', '')[:100] + '...' if len(doc.get('content', '')) > 100 else doc.get('content', ''),
                'length': len(doc.get('content', '')),
                'contains_target': '2.7' in doc.get('content', '') and 'Maximum' in doc.get('content', '')  # 标识是否包含目标信息
            },
            
            # 文档定位数据（用于前端跳转）
            'location_data': {
                'file_path': f"/docs/{doc.get('filename', 'unknown')}",
                'chunk_id': doc.get('chunk_id'),
                'start_line': doc.get('start_line'),
                'end_line': doc.get('end_line'),
                'start_char': doc.get('start_char', 0),
                'end_char': doc.get('end_char', 0),
                'anchor_text': doc.get('content', '')[:50].replace('\n', ' ') + '...' if doc.get('content') else '',
                # 确保前端使用正确的内容进行高亮
                'highlight_content': doc.get('content', ''),  # 用于高亮的完整内容
                'content_type': 'processed_chunk'  # 标识这是处理后的分片内容
            },
            
            # 文档预览URL
            'preview_url': f"/api/documents/preview/{doc.get('filename', 'unknown')}?chunk_id={doc.get('chunk_id', '')}&highlight=true",
            
            # 下载链接
            'download_url': f"/api/documents/download/{doc.get('filename', 'unknown')}"
        }
    
    def _build_agentic_system_prompt(self, agentic_result: Dict[str, Any]) -> str:
        """构建Agentic系统提示词"""
        query_type = agentic_result['agentic_analysis']['query_type']
        sub_questions = agentic_result['agentic_analysis']['sub_questions']
        reasoning_steps = agentic_result['agentic_analysis']['reasoning_steps']
        
        base_prompt = f"""你是一个专业的法律法规智能助手，专门回答与法律条文、技术标准和规范相关的问题。

## 查询分析结果
- 查询类型: {query_type}
- 子问题数量: {len(sub_questions)}
- 推理步骤: {len(reasoning_steps)}

## 🚨 严格引用规范 - 必须遵守
1. **内容真实性**：只能引用提供文档中真实存在的内容，禁止编造或推测
2. **引用格式**：使用格式【文档：filename.md，片段ID：chunk_123，相似度：0.85】
3. **内容匹配**：引用的内容必须与对应片段ID中的实际内容完全匹配
4. **禁止幻觉**：如果某个片段不包含相关信息，不得强行关联或声称它包含相关内容

## 回答要求
1. **引用准确性**：每个事实都应有明确的文档来源，且内容必须真实存在于该文档片段中
2. **结构清晰**：使用分点、分段的方式组织答案
3. **逻辑完整**：按照推理步骤循序渐进地回答
4. **专业术语**：准确使用法律和技术术语
5. **不确定性声明**：如果信息不完整或找不到相关内容，明确指出限制

## ⚠️ 特别注意
- 如果你不能在提供的文档片段中找到确切的信息，请明确说明"在提供的文档中未找到相关信息"
- 不要基于文档片段ID或文件名推测内容，只能基于实际的文档内容进行回答
- 引用时必须确保引用的文字确实出现在对应的文档片段中

## 推理路径
{chr(10).join([f"步骤{i}: {step['description']}" for i, step in enumerate(reasoning_steps, 1)])}

请基于上述分析和提供的文档内容，给出专业、准确、结构化的回答。"""
        
        return base_prompt
    
    def query_with_conversation(
        self, 
        question: str, 
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        带对话状态的查询（多轮对话版本）
        
        Args:
            question: 用户问题
            session_id: 会话ID（可选）
            user_id: 用户ID（可选）
            **kwargs: 其他参数
            
        Returns:
            查询结果（包含对话信息）
        """
        logger.info(f"多轮对话查询: {question}")
        start_time = time.time()
        
        try:
            # 1. 处理用户输入（多轮对话）
            conversation_result = self.multi_turn_processor.process_user_input(
                user_input=question,
                session_id=session_id,
                user_id=user_id
            )
            
            session_id = conversation_result["session_id"]
            
            # 2. 如果需要澄清，直接返回澄清请求
            if conversation_result["needs_clarification"]:
                end_time = time.time()
                return {
                    'answer': conversation_result["clarification"],
                    'confidence': 0.9,
                    'sources': [],
                    'query_id': str(uuid.uuid4()),
                    'response_time': end_time - start_time,
                    'session_id': session_id,
                    'conversation_type': 'clarification',
                    'needs_user_response': True
                }
            
            # 3. 使用对话上下文增强查询
            enhanced_question = self.multi_turn_processor.enhance_query_with_context(
                question, 
                conversation_result["conversation_context"]
            )
            
            # 4. 执行标准RAG流程（使用增强查询）
            rag_result = self.query(enhanced_question, **kwargs)
            
            # 5. 处理工具调用（如果需要）
            tool_result = self.tool_manager.process_with_tools(
                question=question,
                initial_response=rag_result['answer'],
                context=rag_result.get('sources', [])
            )
            
            # 6. 如果有工具调用，增强回答
            if tool_result["has_tool_calls"]:
                enhanced_context = tool_result["enhanced_context"]
                
                # 重新生成带工具结果的回答
                if self.llm_client:
                    enhanced_prompt = f"""
原始回答：{rag_result['answer']}

工具调用结果：
{enhanced_context}

请基于工具调用结果来增强和完善你的回答：
"""
                    enhanced_answer = self.llm_client.generate(enhanced_prompt)
                    rag_result['answer'] = enhanced_answer
                
                rag_result['tool_calls'] = tool_result["tool_calls"]
                rag_result['tool_results'] = tool_result["tool_results"]
            
            # 7. 保存助手回答到对话历史
            self.multi_turn_processor.finalize_response(
                session_id=session_id,
                response=rag_result['answer'],
                metadata={
                    'query_type': rag_result.get('query_type'),
                    'confidence': rag_result.get('confidence'),
                    'sources_count': len(rag_result.get('sources', []))
                },
                tool_calls=tool_result.get("tool_calls"),
                tool_results=tool_result.get("tool_results")
            )
            
            # 8. 返回增强结果
            rag_result.update({
                'session_id': session_id,
                'conversation_type': 'normal',
                'has_conversation_context': bool(conversation_result["conversation_context"]),
                'has_tool_calls': tool_result["has_tool_calls"],
                'needs_user_response': False
            })
            
            return rag_result
            
        except Exception as e:
            logger.error(f"多轮对话查询失败: {e}")
            end_time = time.time()
            
            # 确保返回session_id
            if 'session_id' not in locals():
                session_id = self.conversation_manager.create_session(user_id)
            
            return {
                'answer': f"抱歉，处理您的问题时出现错误: {str(e)}",
                'confidence': 0.1,
                'sources': [],
                'query_id': str(uuid.uuid4()),
                'response_time': end_time - start_time,
                'session_id': session_id,
                'conversation_type': 'error',
                'error': str(e)
            }
    
    def get_conversation_history(self, session_id: str, limit: int = 20) -> List[Dict[str, Any]]:
        """获取对话历史"""
        messages = self.conversation_manager.get_conversation_history(session_id, limit)
        return [msg.to_dict() for msg in messages]
    
    def list_available_tools(self) -> List[Dict[str, Any]]:
        """列出可用工具"""
        return self.tool_manager.registry.list_tools()
    
    def get_conversation_stats(self) -> Dict[str, Any]:
        """获取对话统计信息"""
        return self.conversation_manager.get_conversation_stats() 

    def _build_context(self, context: List[Dict[str, Any]]) -> str:
        """
        构建上下文文本
        
        Args:
            context: 上下文文档列表
            
        Returns:
            格式化的上下文文本
        """
        if not context:
            return "未找到相关文档。"
        
        context_parts = ["以下是相关的参考文档："]
        
        for i, doc in enumerate(context, 1):
            content = doc.get('content', '')
            filename = doc.get('filename', 'unknown')
            chunk_id = doc.get('chunk_id', f'chunk_{i}')
            similarity = doc.get('similarity', 0.0)
            
            # 添加明确的文档标识
            context_parts.append(f"""
【文档{i}：{filename}，片段ID：{chunk_id}，相似度：{similarity:.3f}】
{content}
""")
        
        return "\n".join(context_parts)

    def _build_prompt(self, question: str, context: str, system_prompt: str = None) -> str:
        """
        构建完整的提示词
        
        Args:
            question: 用户问题
            context: 上下文文本
            system_prompt: 系统提示词
            
        Returns:
            完整的提示词
        """
        if system_prompt is None:
            system_prompt = """你是一个专业的法律法规智能助手。请基于提供的文档内容，准确、详细地回答用户的问题。

回答格式：
1. 首先在<think>...</think>标签内进行分析推理
2. 然后提供最终回答

要求：
1. 仅基于提供的文档内容回答，不要编造信息
2. 如果文档中没有相关信息，请明确说明
3. 回答要准确、专业、易懂
4. 可以适当引用文档中的具体条款，格式为【文档：filename，片段ID：chunk_id，相似度：0.xxx】
5. 保持客观中立的态度"""
        
        # 完整提示词
        prompt = f"""{system_prompt}

{context}

=== 用户问题 ===
{question}

=== 回答 ===
请基于上述文档内容回答用户问题："""
        
        return prompt 
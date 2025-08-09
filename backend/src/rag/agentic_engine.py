"""
Agentic RAG引擎

实现智能代理式的检索增强生成，具备规划、执行、反思能力。
"""

import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from src.utils.config import Config
from src.utils.logger import get_logger

logger = get_logger(__name__)


class QueryType(Enum):
    """查询类型"""
    FACTUAL = "factual"          # 事实性查询
    ANALYTICAL = "analytical"    # 分析性查询
    COMPARATIVE = "comparative"  # 比较性查询
    PROCEDURAL = "procedural"    # 程序性查询
    DEFINITIONAL = "definitional" # 定义性查询


@dataclass
class QueryPlan:
    """查询计划"""
    original_question: str
    query_type: QueryType
    sub_questions: List[str]
    search_strategies: List[str]
    reasoning_steps: List[str]
    confidence: float


@dataclass
class ReasoningStep:
    """推理步骤"""
    step_id: int
    description: str
    evidence: List[Dict[str, Any]]
    conclusion: str
    confidence: float


class QueryAnalyzer:
    """查询分析器 - 分析问题类型和复杂度"""
    
    def __init__(self):
        # 问题类型识别模式
        self.patterns = {
            QueryType.DEFINITIONAL: [
                r"什么是",
                r"如何定义",
                r".*的含义",
                r".*是指什么",
                r".*的概念"
            ],
            QueryType.FACTUAL: [
                r".*多少",
                r".*什么时候",
                r".*在哪里",
                r".*是否",
                r".*有没有"
            ],
            QueryType.ANALYTICAL: [
                r"为什么",
                r".*的原因",
                r".*如何影响",
                r".*的作用",
                r"分析.*"
            ],
            QueryType.COMPARATIVE: [
                r".*和.*的区别",
                r".*与.*相比",
                r".*哪个",
                r"比较.*",
                r".*优缺点"
            ],
            QueryType.PROCEDURAL: [
                r"如何.*",
                r"怎样.*",
                r".*的步骤",
                r".*的流程",
                r".*怎么办"
            ]
        }
    
    def analyze_query(self, question: str) -> QueryType:
        """
        分析查询类型
        
        Args:
            question: 用户问题
            
        Returns:
            查询类型
        """
        question = question.strip()
        
        for query_type, patterns in self.patterns.items():
            for pattern in patterns:
                if re.search(pattern, question):
                    logger.debug(f"识别查询类型: {query_type.value}")
                    return query_type
        
        # 默认为事实性查询
        return QueryType.FACTUAL


class QueryPlanner:
    """查询规划器 - 制定查询执行计划"""
    
    def __init__(self):
        self.analyzer = QueryAnalyzer()
    
    def plan_query(self, question: str) -> QueryPlan:
        """
        制定查询计划
        
        Args:
            question: 用户问题
            
        Returns:
            查询计划
        """
        query_type = self.analyzer.analyze_query(question)
        
        # 根据查询类型制定计划
        if query_type == QueryType.DEFINITIONAL:
            return self._plan_definitional(question, query_type)
        elif query_type == QueryType.ANALYTICAL:
            return self._plan_analytical(question, query_type)
        elif query_type == QueryType.COMPARATIVE:
            return self._plan_comparative(question, query_type)
        elif query_type == QueryType.PROCEDURAL:
            return self._plan_procedural(question, query_type)
        else:
            return self._plan_factual(question, query_type)
    
    def _plan_definitional(self, question: str, query_type: QueryType) -> QueryPlan:
        """定义性查询计划"""
        return QueryPlan(
            original_question=question,
            query_type=query_type,
            sub_questions=[
                question,
                f"请详细解释{self._extract_key_term(question)}",
                f"{self._extract_key_term(question)}的特点是什么"
            ],
            search_strategies=["exact_match", "semantic_search"],
            reasoning_steps=[
                "提取关键概念定义",
                "收集相关特征描述",
                "整合完整定义"
            ],
            confidence=0.8
        )
    
    def _plan_analytical(self, question: str, query_type: QueryType) -> QueryPlan:
        """分析性查询计划"""
        return QueryPlan(
            original_question=question,
            query_type=query_type,
            sub_questions=[
                question,
                f"这涉及哪些方面的因素",
                f"这些因素如何相互作用"
            ],
            search_strategies=["semantic_search", "multi_document"],
            reasoning_steps=[
                "识别关键因素",
                "分析因果关系",
                "构建逻辑链条",
                "得出综合结论"
            ],
            confidence=0.6
        )
    
    def _plan_comparative(self, question: str, query_type: QueryType) -> QueryPlan:
        """比较性查询计划"""
        entities = self._extract_comparison_entities(question)
        
        sub_questions = [question]
        for entity in entities:
            sub_questions.append(f"{entity}的特点是什么")
        
        return QueryPlan(
            original_question=question,
            query_type=query_type,
            sub_questions=sub_questions,
            search_strategies=["entity_focused", "parallel_search"],
            reasoning_steps=[
                "分别收集各实体信息",
                "识别比较维度",
                "逐一对比分析",
                "总结异同点"
            ],
            confidence=0.7
        )
    
    def _plan_procedural(self, question: str, query_type: QueryType) -> QueryPlan:
        """程序性查询计划"""
        return QueryPlan(
            original_question=question,
            query_type=query_type,
            sub_questions=[
                question,
                f"这个过程包含哪些步骤",
                f"每个步骤的具体要求是什么"
            ],
            search_strategies=["procedural_search", "step_by_step"],
            reasoning_steps=[
                "识别主要步骤",
                "确定步骤顺序",
                "补充细节要求",
                "组织完整流程"
            ],
            confidence=0.8
        )
    
    def _plan_factual(self, question: str, query_type: QueryType) -> QueryPlan:
        """事实性查询计划"""
        return QueryPlan(
            original_question=question,
            query_type=query_type,
            sub_questions=[question],
            search_strategies=["exact_match"],
            reasoning_steps=[
                "定位相关信息",
                "验证信息准确性",
                "提供直接答案"
            ],
            confidence=0.9
        )
    
    def _extract_key_term(self, question: str) -> str:
        """提取问题中的关键术语"""
        # 简单的关键词提取
        patterns = [
            r"什么是(.+?)[\?？]",
            r"(.+?)是什么",
            r"(.+?)的含义",
            r"(.+?)的概念"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, question)
            if match:
                return match.group(1).strip()
        
        return "相关概念"
    
    def _extract_comparison_entities(self, question: str) -> List[str]:
        """提取比较实体"""
        # 简单的实体提取
        patterns = [
            r"(.+?)和(.+?)的区别",
            r"(.+?)与(.+?)相比",
            r"比较(.+?)和(.+?)"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, question)
            if match:
                return [match.group(1).strip(), match.group(2).strip()]
        
        return []


class ReasoningEngine:
    """推理引擎 - 多步骤逻辑推理"""
    
    def __init__(self):
        pass
    
    def reason_step_by_step(
        self, 
        plan: QueryPlan, 
        retrieved_docs: List[Dict[str, Any]]
    ) -> List[ReasoningStep]:
        """
        逐步推理
        
        Args:
            plan: 查询计划
            retrieved_docs: 检索到的文档
            
        Returns:
            推理步骤列表
        """
        reasoning_steps = []
        
        for i, step_desc in enumerate(plan.reasoning_steps):
            # 为每个推理步骤分配相关证据
            relevant_docs = self._select_relevant_docs(step_desc, retrieved_docs)
            
            # 生成推理步骤
            conclusion = self._generate_step_conclusion(step_desc, relevant_docs)
            
            step = ReasoningStep(
                step_id=i + 1,
                description=step_desc,
                evidence=relevant_docs,
                conclusion=conclusion,
                confidence=self._calculate_step_confidence(relevant_docs)
            )
            
            reasoning_steps.append(step)
        
        return reasoning_steps
    
    def _select_relevant_docs(self, step_desc: str, docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """为推理步骤选择相关文档"""
        # 简单实现：返回前3个最相关的文档
        return docs[:3]
    
    def _generate_step_conclusion(self, step_desc: str, docs: List[Dict[str, Any]]) -> str:
        """为推理步骤生成结论"""
        if not docs:
            return f"基于 '{step_desc}'，但缺少相关证据。"
        
        # 简单实现：基于文档内容生成结论
        doc_contents = [doc.get('content', '')[:200] for doc in docs]
        return f"基于 '{step_desc}'，从 {len(docs)} 个文档中分析得出相关信息。"
    
    def _calculate_step_confidence(self, docs: List[Dict[str, Any]]) -> float:
        """计算推理步骤的置信度"""
        if not docs:
            return 0.1
        
        # 基于文档数量和平均相似度
        avg_similarity = sum(doc.get('similarity', 0) for doc in docs) / len(docs)
        return min(0.9, avg_similarity * 1.2)


class AgenticRAGEngine:
    """Agentic RAG引擎 - 智能代理式检索增强生成"""
    
    def __init__(self, config: Config):
        """
        初始化Agentic RAG引擎
        
        Args:
            config: 配置对象
        """
        self.config = config
        self.planner = QueryPlanner()
        self.reasoning_engine = ReasoningEngine()
        
        logger.info("Agentic RAG引擎初始化完成")
    
    def process_query(
        self, 
        question: str, 
        retrieved_docs: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        处理查询 - Agentic RAG主流程
        
        Args:
            question: 用户问题
            retrieved_docs: 检索到的文档
            
        Returns:
            增强的查询结果
        """
        logger.info(f"Agentic RAG处理查询: {question}")
        
        # 1. 查询规划
        plan = self.planner.plan_query(question)
        logger.info(f"查询计划: {plan.query_type.value}, {len(plan.sub_questions)} 个子问题")
        
        # 详细记录子问题
        for i, sub_q in enumerate(plan.sub_questions, 1):
            logger.info(f"  子问题{i}: {sub_q}")
        
        # 2. 多步推理
        reasoning_steps = self.reasoning_engine.reason_step_by_step(plan, retrieved_docs)
        logger.info(f"推理完成: {len(reasoning_steps)} 个步骤")
        
        # 详细记录推理步骤
        for step in reasoning_steps:
            logger.info(f"  步骤{step.step_id}: {step.description} -> {step.conclusion[:100]}...")
        
        # 3. 生成增强结果
        enhanced_result = self._generate_enhanced_response(plan, reasoning_steps, retrieved_docs)
        
        return enhanced_result
    
    def _generate_enhanced_response(
        self, 
        plan: QueryPlan, 
        reasoning_steps: List[ReasoningStep],
        retrieved_docs: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """生成增强的响应"""
        
        # 构建推理链条文本
        reasoning_chain = "\n".join([
            f"步骤{step.step_id}: {step.description} -> {step.conclusion}"
            for step in reasoning_steps
        ])
        
        # 计算总体置信度
        overall_confidence = sum(step.confidence for step in reasoning_steps) / len(reasoning_steps) if reasoning_steps else 0.5
        
        return {
            'agentic_analysis': {
                'query_type': plan.query_type.value,
                'sub_questions': plan.sub_questions,
                'reasoning_steps': [
                    {
                        'step': step.step_id,
                        'description': step.description,
                        'conclusion': step.conclusion,
                        'confidence': step.confidence,
                        'evidence_count': len(step.evidence)
                    }
                    for step in reasoning_steps
                ],
                'reasoning_chain': reasoning_chain,
                'overall_confidence': overall_confidence
            },
            'enhanced_context': self._build_enhanced_context(plan, retrieved_docs),
            'search_strategies': plan.search_strategies
        }
    
    def _build_enhanced_context(self, plan: QueryPlan, docs: List[Dict[str, Any]]) -> str:
        """构建增强的上下文"""
        if not docs:
            return "未找到相关文档。"
        
        context_parts = []
        
        # 添加查询类型说明
        context_parts.append(f"查询类型: {plan.query_type.value}")
        
        # 根据查询类型组织上下文
        if plan.query_type == QueryType.COMPARATIVE:
            context_parts.append("以下是用于比较分析的相关信息:")
        elif plan.query_type == QueryType.ANALYTICAL:
            context_parts.append("以下是用于深入分析的相关信息:")
        elif plan.query_type == QueryType.PROCEDURAL:
            context_parts.append("以下是相关的步骤和流程信息:")
        else:
            context_parts.append("以下是相关的背景信息:")
        
        # 添加文档内容
        for i, doc in enumerate(docs[:5], 1):
            content = doc.get('content', '')[:300]
            similarity = doc.get('similarity', 0)
            context_parts.append(f"\n[文档{i} - 相似度:{similarity:.3f}]\n{content}")
        
        return "\n".join(context_parts) 
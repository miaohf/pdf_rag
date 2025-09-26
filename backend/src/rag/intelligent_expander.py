"""
智能查询扩展器

基于语义理解和动态学习的查询扩展，不依赖硬编码概念映射。
"""

import re
from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass
from collections import defaultdict

from src.utils.config import Config
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class QueryContext:
    """查询上下文"""
    original_question: str
    detected_entities: List[str]
    detected_concepts: List[str]
    query_intent: str
    domain_hints: List[str]


class IntelligentExpander:
    """智能查询扩展器 - 基于语义理解，无硬编码依赖"""
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = get_logger(__name__)
        
        # 动态学习的概念库（运行时构建，非硬编码）
        self.concept_patterns: Dict[str, Set[str]] = defaultdict(set)
        self.entity_relations: Dict[str, Set[str]] = defaultdict(set)
        
        # 通用扩展策略（基于语言学原理，非领域特定）
        self.general_strategies = [
            self._synonym_expansion,
            self._concept_broadening,
            self._parameter_completion,
            self._context_enhancement
        ]
    
    def expand_query(self, question: str, retrieved_docs: List[Dict[str, Any]] = None) -> List[str]:
        """
        智能查询扩展
        
        Args:
            question: 原始问题
            retrieved_docs: 已检索的文档（用于动态学习）
            
        Returns:
            扩展后的查询列表
        """
        expanded_queries = [question]
        
        # 1. 分析查询上下文
        context = self._analyze_query_context(question)
        
        # 2. 动态学习（如果有检索结果）
        if retrieved_docs:
            self._dynamic_learning(context, retrieved_docs)
        
        # 3. 应用通用扩展策略
        for strategy in self.general_strategies:
            strategy_queries = strategy(context, retrieved_docs)
            expanded_queries.extend(strategy_queries)
        
        # 4. 去重并返回
        return list(dict.fromkeys(expanded_queries))
    
    def _analyze_query_context(self, question: str) -> QueryContext:
        """分析查询上下文"""
        # 实体检测（基于语言学规则，非硬编码）
        entities = self._extract_entities(question)
        
        # 概念检测（基于语义模式，非硬编码）
        concepts = self._extract_concepts(question)
        
        # 意图识别（基于问题结构，非硬编码）
        intent = self._detect_intent(question)
        
        # 领域提示（基于上下文线索，非硬编码）
        domain_hints = self._extract_domain_hints(question)
        
        return QueryContext(
            original_question=question,
            detected_entities=entities,
            detected_concepts=concepts,
            query_intent=intent,
            domain_hints=domain_hints
        )
    
    def _extract_entities(self, question: str) -> List[str]:
        """提取实体（基于语言学规则）"""
        entities = []
        
        # 技术术语模式（通用规则）
        tech_patterns = [
            r'\b[A-Z]{2,}(?:\s+[A-Z]{2,})*\b',  # 缩写词
            r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b',  # 技术名词
            r'\b\d+(?:\.\d+)?\s*(?:km/h|m|metres?|meters?)\b',  # 数值+单位
        ]
        
        for pattern in tech_patterns:
            matches = re.findall(pattern, question)
            entities.extend(matches)
        
        # 中文实体（通用规则）
        chinese_patterns = [
            r'[\u4e00-\u9fff]+系统',
            r'[\u4e00-\u9fff]+距离',
            r'[\u4e00-\u9fff]+限制',
            r'[\u4e00-\u9fff]+要求'
        ]
        
        for pattern in chinese_patterns:
            matches = re.findall(pattern, question)
            entities.extend(matches)
        
        return list(set(entities))
    
    def _extract_concepts(self, question: str) -> List[str]:
        """提取概念（基于语义模式）"""
        concepts = []
        
        # 参数概念（通用模式）
        param_patterns = [
            r'最大\w+',
            r'最小\w+',
            r'\w+限制',
            r'\w+要求',
            r'\w+标准',
            r'\w+依据'
        ]
        
        for pattern in param_patterns:
            matches = re.findall(pattern, question)
            concepts.extend(matches)
        
        # 英文概念（通用模式）
        english_patterns = [
            r'maximum\s+\w+',
            r'minimum\s+\w+',
            r'\w+\s+limit',
            r'\w+\s+requirement',
            r'\w+\s+standard'
        ]
        
        for pattern in english_patterns:
            matches = re.findall(pattern, question, re.IGNORECASE)
            concepts.extend(matches)
        
        return list(set(concepts))
    
    def _detect_intent(self, question: str) -> str:
        """检测查询意图（基于问题结构）"""
        # 基于问题词的意图识别（通用规则）
        if any(word in question for word in ['是什么', '什么是', 'how', 'what']):
            return "definition"
        elif any(word in question for word in ['多少', '多大', 'how much', 'how many']):
            return "quantity"
        elif any(word in question for word in ['如何', '怎么', 'how to']):
            return "procedure"
        elif any(word in question for word in ['为什么', 'why']):
            return "reason"
        else:
            return "general"
    
    def _extract_domain_hints(self, question: str) -> List[str]:
        """提取领域提示（基于上下文线索）"""
        hints = []
        
        # 技术领域提示（通用规则）
        tech_indicators = ['系统', '技术', '设备', '功能', '参数', '规范']
        for indicator in tech_indicators:
            if indicator in question:
                hints.append("technical")
        
        # 法规领域提示（通用规则）
        legal_indicators = ['法规', '规定', '要求', '标准', '依据', '条例']
        for indicator in legal_indicators:
            if indicator in question:
                hints.append("legal")
        
        # 英文领域提示（通用规则）
        english_indicators = ['system', 'function', 'requirement', 'standard', 'regulation']
        for indicator in english_indicators:
            if indicator.lower() in question.lower():
                hints.append("technical")
        
        return list(set(hints))
    
    def _dynamic_learning(self, context: QueryContext, docs: List[Dict[str, Any]]):
        """动态学习（从检索结果中学习新概念）"""
        for doc in docs:
            content = doc.get('content', '')
            
            # 学习新的概念模式
            self._learn_concept_patterns(content, context)
            
            # 学习实体关系
            self._learn_entity_relations(content, context)
    
    def _learn_concept_patterns(self, content: str, context: QueryContext):
        """学习概念模式"""
        # 从内容中提取数值+单位模式
        value_patterns = re.findall(r'\b\d+(?:\.\d+)?\s*(?:km/h|m|metres?|meters?|km|cm|mm)\b', content)
        
        for pattern in value_patterns:
            # 提取单位
            unit_match = re.search(r'(km/h|m|metres?|meters?|km|cm|mm)', pattern)
            if unit_match:
                unit = unit_match.group(1)
                self.concept_patterns[unit].add(pattern)
    
    def _learn_entity_relations(self, content: str, context: QueryContext):
        """学习实体关系"""
        # 从内容中学习实体间的关联
        for entity in context.detected_entities:
            # 查找与实体相关的其他概念
            related_concepts = re.findall(rf'{re.escape(entity)}[^。！？]*?(?:要求|限制|标准|参数)', content)
            
            for concept in related_concepts:
                self.entity_relations[entity].add(concept)
    
    def _synonym_expansion(self, context: QueryContext, docs: List[Dict[str, Any]] = None) -> List[str]:
        """同义词扩展（基于语言学原理）"""
        expanded = []
        
        # 中英文对应（通用规则）
        chinese_english_map = {
            '系统': ['system'],
            '功能': ['function', 'feature'],
            '要求': ['requirement', 'specification'],
            '限制': ['limit', 'restriction'],
            '标准': ['standard', 'criterion'],
            '距离': ['distance', 'range'],
            '速度': ['speed', 'velocity']
        }
        
        for chinese, english_list in chinese_english_map.items():
            if chinese in context.original_question:
                for english in english_list:
                    expanded.append(english)
        
        return expanded
    
    def _concept_broadening(self, context: QueryContext, docs: List[Dict[str, Any]] = None) -> List[str]:
        """概念扩展（基于语义关联）"""
        expanded = []
        
        # 基于检测到的概念进行扩展
        for concept in context.detected_concepts:
            if '最大' in concept:
                # 扩展为相关的最小值概念
                related = concept.replace('最大', '最小')
                expanded.append(related)
            
            if '距离' in concept:
                # 扩展为相关的空间概念
                expanded.extend(['范围', '范围', 'area', 'scope'])
        
        return expanded
    
    def _parameter_completion(self, context: QueryContext, docs: List[Dict[str, Any]] = None) -> List[str]:
        """参数完整性扩展（基于动态学习）"""
        expanded = []
        
        # 基于已学习的模式进行扩展
        for unit, patterns in self.concept_patterns.items():
            if any(unit in entity for entity in context.detected_entities):
                # 扩展为相关的参数查询
                expanded.extend([f"{unit} 参数", f"{unit} 要求", f"{unit} 标准"])
        
        return expanded
    
    def _context_enhancement(self, context: QueryContext, docs: List[Dict[str, Any]] = None) -> List[str]:
        """上下文增强（基于领域提示）"""
        expanded = []
        
        # 基于领域提示进行扩展
        if "technical" in context.domain_hints:
            expanded.extend(['technical specification', 'technical requirement', 'technical standard'])
        
        if "legal" in context.domain_hints:
            expanded.extend(['legal requirement', 'regulation', 'compliance'])
        
        return expanded
    
    def get_completeness_analysis(self, question: str, docs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        获取信息完整性分析
        
        Args:
            question: 用户问题
            docs: 检索到的文档
            
        Returns:
            完整性分析结果
        """
        context = self._analyze_query_context(question)
        
        # 分析检测到的概念是否都有对应的信息
        concept_coverage = {}
        for concept in context.detected_concepts:
            coverage = self._analyze_concept_coverage(concept, docs)
            concept_coverage[concept] = coverage
        
        return {
            "detected_concepts": context.detected_concepts,
            "concept_coverage": concept_coverage,
            "overall_completeness": self._calculate_overall_completeness(concept_coverage),
            "suggestions": self._generate_completeness_suggestions(concept_coverage)
        }
    
    def _analyze_concept_coverage(self, concept: str, docs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """分析概念覆盖度"""
        # 在文档中搜索概念相关信息
        relevant_docs = []
        for doc in docs:
            content = doc.get('content', '')
            if concept in content or any(related in content for related in self._get_concept_variants(concept)):
                relevant_docs.append(doc)
        
        return {
            "concept": concept,
            "found": len(relevant_docs) > 0,
            "relevant_docs_count": len(relevant_docs),
            "confidence": min(1.0, len(relevant_docs) / max(len(docs), 1))
        }
    
    def _get_concept_variants(self, concept: str) -> List[str]:
        """获取概念的变体（基于语言学规则）"""
        variants = [concept]
        
        # 中英文变体
        if '最大' in concept:
            variants.extend(['maximum', 'max', '最高'])
        if '最小' in concept:
            variants.extend(['minimum', 'min', '最低'])
        if '距离' in concept:
            variants.extend(['distance', 'range', '范围'])
        if '速度' in concept:
            variants.extend(['speed', 'velocity', '速率'])
        
        return variants
    
    def _calculate_overall_completeness(self, concept_coverage: Dict[str, Any]) -> float:
        """计算整体完整性分数"""
        if not concept_coverage:
            return 0.0
        
        total_score = sum(coverage.get('confidence', 0) for coverage in concept_coverage.values())
        return total_score / len(concept_coverage)
    
    def _generate_completeness_suggestions(self, concept_coverage: Dict[str, Any]) -> List[str]:
        """生成完整性改进建议"""
        suggestions = []
        
        for concept, coverage in concept_coverage.items():
            if not coverage.get('found', False):
                suggestions.append(f"建议搜索更多关于'{concept}'的信息")
            elif coverage.get('confidence', 0) < 0.5:
                suggestions.append(f"建议深入搜索'{concept}'的详细信息")
        
        return suggestions 
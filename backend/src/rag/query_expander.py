"""
智能查询扩展器

基于配置和知识图谱的查询扩展，避免硬编码关键词映射。
"""

import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from pathlib import Path
import yaml

from src.utils.config import Config
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ConceptMapping:
    """概念映射"""
    concept: str
    synonyms: List[str]
    related_concepts: List[str]
    domain: str


@dataclass
class ParameterMapping:
    """参数映射"""
    parameter: str
    keywords: List[str]
    value_patterns: List[str]
    unit: Optional[str] = None


class QueryExpander:
    """智能查询扩展器"""
    
    def __init__(self, config: Config):
        self.config = config
        self.concept_mappings: Dict[str, ConceptMapping] = {}
        self.parameter_mappings: Dict[str, ParameterMapping] = {}
        self.domain_configs: Dict[str, Dict[str, Any]] = {}
        
        self._load_configurations()
    
    def _load_configurations(self):
        """加载配置文件"""
        # 从配置文件加载
        self.domain_configs = self.config.retrieval.domain_configs
        
        # 从外部配置文件加载（如果存在）
        config_path = Path("config/query_expansion.yaml")
        if config_path.exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                external_config = yaml.safe_load(f)
                self._merge_configurations(external_config)
        
        logger.info(f"加载了 {len(self.domain_configs)} 个领域配置")
    
    def _merge_configurations(self, external_config: Dict[str, Any]):
        """合并外部配置"""
        if "domain_configs" in external_config:
            for domain, config in external_config["domain_configs"].items():
                if domain in self.domain_configs:
                    self.domain_configs[domain].update(config)
                else:
                    self.domain_configs[domain] = config
    
    def expand_query(self, question: str, domain: Optional[str] = None) -> List[str]:
        """
        智能查询扩展
        
        Args:
            question: 原始问题
            domain: 领域标识
            
        Returns:
            扩展后的查询列表
        """
        expanded_queries = [question]
        
        # 自动检测领域
        if not domain:
            domain = self._detect_domain(question)
        
        if domain and domain in self.domain_configs:
            domain_config = self.domain_configs[domain]
            expanded_queries.extend(self._expand_by_domain(question, domain_config))
        
        # 通用扩展策略
        expanded_queries.extend(self._apply_general_strategies(question))
        
        # 去重并返回
        return list(dict.fromkeys(expanded_queries))
    
    def _detect_domain(self, question: str) -> Optional[str]:
        """自动检测问题领域"""
        for domain, config in self.domain_configs.items():
            core_concepts = config.get("core_concepts", [])
            if any(concept.lower() in question.lower() for concept in core_concepts):
                return domain
        return None
    
    def _expand_by_domain(self, question: str, domain_config: Dict[str, Any]) -> List[str]:
        """基于领域配置的查询扩展"""
        expanded = []
        
        # 核心概念扩展
        core_concepts = domain_config.get("core_concepts", [])
        if core_concepts:
            expanded.extend(core_concepts)
        
        # 参数相关扩展
        key_parameters = domain_config.get("key_parameters", [])
        parameter_mappings = domain_config.get("parameter_mappings", {})
        
        for param in key_parameters:
            if param in question:
                mappings = parameter_mappings.get(param, [])
                expanded.extend(mappings)
        
        return expanded
    
    def _apply_general_strategies(self, question: str) -> List[str]:
        """应用通用扩展策略"""
        expanded = []
        
        # 同义词扩展（基于配置）
        # 概念关联扩展（基于配置）
        # 领域术语扩展（基于配置）
        
        return expanded
    
    def get_missing_parameters(self, question: str, retrieved_docs: List[Dict[str, Any]], domain: str) -> List[str]:
        """
        检查缺失的参数信息
        
        Args:
            question: 用户问题
            retrieved_docs: 检索到的文档
            domain: 领域标识
            
        Returns:
            缺失的参数列表
        """
        if domain not in self.domain_configs:
            return []
        
        domain_config = self.domain_configs[domain]
        key_parameters = domain_config.get("key_parameters", [])
        parameter_mappings = domain_config.get("parameter_mappings", {})
        
        missing_params = []
        
        for param in key_parameters:
            if param in question:  # 用户询问了这个参数
                param_found = False
                mappings = parameter_mappings.get(param, [])
                
                # 检查文档中是否包含该参数信息
                for doc in retrieved_docs:
                    content = doc.get("content", "").lower()
                    if any(mapping.lower() in content for mapping in mappings):
                        param_found = True
                        break
                
                if not param_found:
                    missing_params.append(param)
        
        return missing_params
    
    def suggest_additional_queries(self, missing_params: List[str], domain: str) -> List[str]:
        """
        为缺失的参数建议额外的查询
        
        Args:
            missing_params: 缺失的参数列表
            domain: 领域标识
            
        Returns:
            建议的查询列表
        """
        if domain not in self.domain_configs:
            return []
        
        domain_config = self.domain_configs[domain]
        parameter_mappings = domain_config.get("parameter_mappings", {})
        
        suggested_queries = []
        
        for param in missing_params:
            mappings = parameter_mappings.get(param, [])
            if mappings:
                # 生成针对该参数的专门查询
                suggested_queries.extend(mappings)
        
        return suggested_queries 
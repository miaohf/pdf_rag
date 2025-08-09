"""
工具调用系统

为Agentic RAG提供外部工具和API集成能力。
"""

import json
import httpx
import re
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Callable, Union
from dataclasses import dataclass
from datetime import datetime
import asyncio

from src.utils.logger import get_logger
from src.utils.config import Config

logger = get_logger(__name__)


@dataclass
class ToolResult:
    """工具执行结果"""
    success: bool
    result: Any
    error: Optional[str] = None
    execution_time: float = 0.0
    tool_name: str = ""
    metadata: Dict[str, Any] = None


class BaseTool(ABC):
    """工具基类"""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
    
    @abstractmethod
    def execute(self, **kwargs) -> ToolResult:
        """执行工具"""
        pass
    
    @abstractmethod
    def get_parameters(self) -> Dict[str, Any]:
        """获取工具参数说明"""
        pass
    
    def validate_parameters(self, **kwargs) -> bool:
        """验证参数"""
        return True


class WebSearchTool(BaseTool):
    """网络搜索工具"""
    
    def __init__(self):
        super().__init__(
            name="web_search",
            description="搜索互联网获取最新信息"
        )
        self.client = httpx.Client(timeout=10.0)
    
    def get_parameters(self) -> Dict[str, Any]:
        return {
            "query": {"type": "string", "description": "搜索查询词", "required": True},
            "num_results": {"type": "integer", "description": "返回结果数量", "default": 5}
        }
    
    def execute(self, query: str, num_results: int = 5) -> ToolResult:
        """执行网络搜索"""
        start_time = datetime.now()
        
        try:
            # 模拟搜索API调用（实际可接入Google、Bing等API）
            logger.info(f"执行网络搜索: {query}")
            
            # 这里可以接入真实的搜索API
            mock_results = [
                {
                    "title": f"搜索结果1 - {query}",
                    "url": f"https://example.com/1?q={query}",
                    "snippet": f"关于{query}的详细信息...",
                    "date": "2024-08-06"
                },
                {
                    "title": f"搜索结果2 - {query}",
                    "url": f"https://example.com/2?q={query}",
                    "snippet": f"更多关于{query}的内容...",
                    "date": "2024-08-05"
                }
            ]
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            return ToolResult(
                success=True,
                result=mock_results[:num_results],
                execution_time=execution_time,
                tool_name=self.name,
                metadata={"query": query, "num_results": num_results}
            )
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            logger.error(f"网络搜索失败: {e}")
            
            return ToolResult(
                success=False,
                result=None,
                error=str(e),
                execution_time=execution_time,
                tool_name=self.name
            )


class CalculatorTool(BaseTool):
    """计算器工具"""
    
    def __init__(self):
        super().__init__(
            name="calculator",
            description="执行数学计算"
        )
    
    def get_parameters(self) -> Dict[str, Any]:
        return {
            "expression": {"type": "string", "description": "数学表达式", "required": True}
        }
    
    def execute(self, expression: str) -> ToolResult:
        """执行数学计算"""
        start_time = datetime.now()
        
        try:
            # 安全的数学表达式计算
            allowed_chars = set('0123456789+-*/.() ')
            if not all(c in allowed_chars for c in expression):
                raise ValueError("表达式包含不允许的字符")
            
            result = eval(expression)
            execution_time = (datetime.now() - start_time).total_seconds()
            
            logger.info(f"计算: {expression} = {result}")
            
            return ToolResult(
                success=True,
                result=result,
                execution_time=execution_time,
                tool_name=self.name,
                metadata={"expression": expression}
            )
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            logger.error(f"计算失败: {e}")
            
            return ToolResult(
                success=False,
                result=None,
                error=str(e),
                execution_time=execution_time,
                tool_name=self.name
            )


class DocumentSearchTool(BaseTool):
    """文档搜索工具（内部知识库）"""
    
    def __init__(self, vector_store=None):
        super().__init__(
            name="document_search",
            description="在内部文档库中搜索相关信息"
        )
        self.vector_store = vector_store
    
    def get_parameters(self) -> Dict[str, Any]:
        return {
            "query": {"type": "string", "description": "搜索查询", "required": True},
            "top_k": {"type": "integer", "description": "返回结果数量", "default": 5}
        }
    
    def execute(self, query: str, top_k: int = 5) -> ToolResult:
        """执行文档搜索"""
        start_time = datetime.now()
        
        try:
            if self.vector_store:
                results = self.vector_store.search_similar(query, top_k=top_k)
            else:
                # 模拟搜索结果
                results = [
                    {
                        "content": f"文档片段1关于{query}的内容...",
                        "filename": "document1.pdf",
                        "similarity": 0.85
                    },
                    {
                        "content": f"文档片段2关于{query}的内容...",
                        "filename": "document2.pdf", 
                        "similarity": 0.78
                    }
                ]
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            return ToolResult(
                success=True,
                result=results,
                execution_time=execution_time,
                tool_name=self.name,
                metadata={"query": query, "results_count": len(results)}
            )
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            logger.error(f"文档搜索失败: {e}")
            
            return ToolResult(
                success=False,
                result=None,
                error=str(e),
                execution_time=execution_time,
                tool_name=self.name
            )


class LegalApiTool(BaseTool):
    """法律法规API工具"""
    
    def __init__(self):
        super().__init__(
            name="legal_api",
            description="查询最新的法律法规信息"
        )
        self.client = httpx.Client(timeout=15.0)
    
    def get_parameters(self) -> Dict[str, Any]:
        return {
            "law_type": {"type": "string", "description": "法律类型", "required": True},
            "keyword": {"type": "string", "description": "关键词", "required": True}
        }
    
    def execute(self, law_type: str, keyword: str) -> ToolResult:
        """查询法律法规"""
        start_time = datetime.now()
        
        try:
            # 模拟法律API调用
            logger.info(f"查询法律法规: {law_type} - {keyword}")
            
            mock_result = {
                "law_name": f"{law_type}相关法律",
                "articles": [
                    {
                        "article_no": "第一条",
                        "content": f"关于{keyword}的规定...",
                        "effective_date": "2024-01-01"
                    },
                    {
                        "article_no": "第二条",
                        "content": f"进一步关于{keyword}的细则...",
                        "effective_date": "2024-01-01"
                    }
                ],
                "last_updated": "2024-08-06"
            }
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            return ToolResult(
                success=True,
                result=mock_result,
                execution_time=execution_time,
                tool_name=self.name,
                metadata={"law_type": law_type, "keyword": keyword}
            )
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            logger.error(f"法律API查询失败: {e}")
            
            return ToolResult(
                success=False,
                result=None,
                error=str(e),
                execution_time=execution_time,
                tool_name=self.name
            )


class ToolRegistry:
    """工具注册表"""
    
    def __init__(self):
        self.tools: Dict[str, BaseTool] = {}
        self._register_default_tools()
    
    def _register_default_tools(self):
        """注册默认工具"""
        self.register_tool(WebSearchTool())
        self.register_tool(CalculatorTool())
        self.register_tool(DocumentSearchTool())
        self.register_tool(LegalApiTool())
        
        logger.info(f"注册了 {len(self.tools)} 个默认工具")
    
    def register_tool(self, tool: BaseTool):
        """注册工具"""
        self.tools[tool.name] = tool
        logger.debug(f"注册工具: {tool.name}")
    
    def get_tool(self, name: str) -> Optional[BaseTool]:
        """获取工具"""
        return self.tools.get(name)
    
    def list_tools(self) -> List[Dict[str, Any]]:
        """列出所有工具"""
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.get_parameters()
            }
            for tool in self.tools.values()
        ]
    
    def execute_tool(self, tool_name: str, **kwargs) -> ToolResult:
        """执行工具"""
        tool = self.get_tool(tool_name)
        if not tool:
            return ToolResult(
                success=False,
                result=None,
                error=f"工具 '{tool_name}' 不存在",
                tool_name=tool_name
            )
        
        try:
            if not tool.validate_parameters(**kwargs):
                return ToolResult(
                    success=False,
                    result=None,
                    error="参数验证失败",
                    tool_name=tool_name
                )
            
            return tool.execute(**kwargs)
            
        except Exception as e:
            logger.error(f"工具执行失败 {tool_name}: {e}")
            return ToolResult(
                success=False,
                result=None,
                error=str(e),
                tool_name=tool_name
            )


class ToolCallParser:
    """工具调用解析器"""
    
    def __init__(self):
        # 工具调用的正则模式
        self.tool_call_pattern = r'<tool_call>\s*(\w+)\((.*?)\)\s*</tool_call>'
        self.json_pattern = r'<tool_call>\s*(\w+)\s*({.*?})\s*</tool_call>'
    
    def parse_tool_calls(self, text: str) -> List[Dict[str, Any]]:
        """
        从文本中解析工具调用
        
        格式支持：
        <tool_call>web_search(query="法律条文", num_results=3)</tool_call>
        <tool_call>calculator(expression="10 + 20")</tool_call>
        """
        tool_calls = []
        
        # 尝试解析标准格式
        matches = re.findall(self.tool_call_pattern, text, re.DOTALL)
        for tool_name, params_str in matches:
            try:
                # 解析参数
                params = self._parse_parameters(params_str)
                tool_calls.append({
                    "tool_name": tool_name,
                    "parameters": params
                })
            except Exception as e:
                logger.warning(f"解析工具调用失败: {tool_name}({params_str}) - {e}")
        
        return tool_calls
    
    def _parse_parameters(self, params_str: str) -> Dict[str, Any]:
        """解析参数字符串"""
        params = {}
        
        if not params_str.strip():
            return params
        
        # 尝试直接作为JSON解析
        try:
            return json.loads(f"{{{params_str}}}")
        except:
            pass
        
        # 解析 key=value 格式
        param_pairs = re.findall(r'(\w+)\s*=\s*([^,]+)', params_str)
        for key, value in param_pairs:
            # 处理字符串值
            if value.startswith('"') and value.endswith('"'):
                params[key] = value[1:-1]
            elif value.startswith("'") and value.endswith("'"):
                params[key] = value[1:-1]
            # 处理数字
            elif value.isdigit():
                params[key] = int(value)
            elif '.' in value and value.replace('.', '').isdigit():
                params[key] = float(value)
            # 处理布尔值
            elif value.lower() in ['true', 'false']:
                params[key] = value.lower() == 'true'
            else:
                params[key] = value
        
        return params
    
    def format_tool_results(self, results: List[ToolResult]) -> str:
        """格式化工具执行结果"""
        if not results:
            return ""
        
        formatted_results = []
        for result in results:
            if result.success:
                formatted_results.append(f"""
<tool_result tool="{result.tool_name}">
{json.dumps(result.result, ensure_ascii=False, indent=2)}
</tool_result>""")
            else:
                formatted_results.append(f"""
<tool_result tool="{result.tool_name}" error="{result.error}">
工具执行失败
</tool_result>""")
        
        return "\n".join(formatted_results)


class ToolManager:
    """工具管理器 - 集成工具调用到RAG流程"""
    
    def __init__(self, config: Config, vector_store=None):
        self.config = config
        self.registry = ToolRegistry()
        self.parser = ToolCallParser()
        
        # 注册文档搜索工具的向量存储
        if vector_store:
            doc_tool = self.registry.get_tool("document_search")
            if doc_tool:
                doc_tool.vector_store = vector_store
        
        logger.info("工具管理器初始化完成")
    
    def should_use_tools(self, question: str, context: List[Dict[str, Any]]) -> bool:
        """判断是否需要使用工具"""
        # 简单的启发式规则
        tool_indicators = [
            "最新", "现在", "目前", "当前",
            "计算", "算一下", "多少",
            "搜索", "查找", "寻找",
            "法律", "法规", "条例"
        ]
        
        # 如果上下文信息不足
        if not context or len(context) < 2:
            return True
        
        # 如果问题包含工具指示词
        for indicator in tool_indicators:
            if indicator in question:
                return True
        
        return False
    
    def suggest_tools(self, question: str, query_type: str) -> List[str]:
        """根据问题和查询类型建议工具"""
        suggested_tools = []
        
        # 根据查询类型建议工具
        if query_type == "factual":
            if any(word in question for word in ["最新", "现在", "目前"]):
                suggested_tools.append("web_search")
            if any(word in question for word in ["法律", "法规", "条例"]):
                suggested_tools.append("legal_api")
        
        elif query_type == "analytical":
            suggested_tools.append("document_search")
            if any(word in question for word in ["计算", "数量", "比例"]):
                suggested_tools.append("calculator")
        
        elif query_type == "comparative":
            suggested_tools.append("document_search")
            suggested_tools.append("web_search")
        
        # 总是建议文档搜索作为备选
        if "document_search" not in suggested_tools:
            suggested_tools.append("document_search")
        
        return suggested_tools
    
    def execute_tool_calls(self, tool_calls: List[Dict[str, Any]]) -> List[ToolResult]:
        """执行工具调用列表"""
        results = []
        
        for call in tool_calls:
            tool_name = call.get("tool_name")
            parameters = call.get("parameters", {})
            
            logger.info(f"执行工具: {tool_name} with {parameters}")
            result = self.registry.execute_tool(tool_name, **parameters)
            results.append(result)
        
        return results
    
    def process_with_tools(self, question: str, initial_response: str, context: List[Dict[str, Any]]) -> Dict[str, Any]:
        """处理带工具调用的响应"""
        # 解析可能的工具调用
        tool_calls = self.parser.parse_tool_calls(initial_response)
        
        if not tool_calls:
            # 如果没有明确的工具调用，但应该使用工具
            if self.should_use_tools(question, context):
                # 自动建议工具调用
                suggested_tools = self.suggest_tools(question, "factual")  # 默认类型
                if suggested_tools:
                    auto_call = {
                        "tool_name": suggested_tools[0],
                        "parameters": {"query": question}
                    }
                    tool_calls = [auto_call]
        
        # 执行工具调用
        tool_results = []
        if tool_calls:
            tool_results = self.execute_tool_calls(tool_calls)
        
        # 格式化结果
        tool_output = self.parser.format_tool_results(tool_results)
        
        return {
            "has_tool_calls": len(tool_calls) > 0,
            "tool_calls": tool_calls,
            "tool_results": tool_results,
            "tool_output": tool_output,
            "enhanced_context": tool_output
        } 
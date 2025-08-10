"""
大语言模型客户端

负责与Ollama LLM的交互和管理。
"""

import httpx
import json
from typing import Dict, Any, Optional, Generator, AsyncGenerator, List
import asyncio
from datetime import datetime

from src.utils.config import Config
from src.utils.logger import get_logger

logger = get_logger(__name__)


class LLMClient:
    """LLM客户端"""
    
    def __init__(self, config: Config):
        """
        初始化LLM客户端
        
        Args:
            config: 配置对象
        """
        self.config = config
        self.ollama_config = config.ollama
        self.api_url = self.ollama_config.url
        self.model_name = self.ollama_config.model
        self.temperature = self.ollama_config.temperature
        self.max_tokens = self.ollama_config.max_tokens
        self.timeout = self.ollama_config.timeout
        
        # HTTP客户端
        self.client = httpx.Client(timeout=self.timeout)
        self.async_client = httpx.AsyncClient(timeout=self.timeout)
        
        logger.info(f"LLM客户端初始化完成: {self.model_name}")
        
        # 验证模型可用性
        self._validate_model()
    
    def _validate_model(self) -> None:
        """验证LLM模型是否可用"""
        try:
            response = self.client.get(f"{self.api_url}/api/tags")
            if response.status_code == 200:
                models = response.json().get('models', [])
                model_names = [m['name'] for m in models]
                
                if self.model_name not in model_names:
                    logger.warning(f"LLM模型 {self.model_name} 未找到，可用模型: {model_names}")
                else:
                    logger.info(f"LLM模型 {self.model_name} 验证成功")
            else:
                logger.warning(f"无法连接到Ollama服务: {response.status_code}")
                
        except Exception as e:
            logger.error(f"验证LLM模型失败: {e}")
    
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
    
    def _build_prompt(self, question: str, context: List[Dict[str, Any]], system_prompt: Optional[str] = None) -> str:
        """
        构建完整的提示词
        
        Args:
            question: 用户问题
            context: 检索到的上下文文档
            system_prompt: 系统提示词
            
        Returns:
            完整的提示词
        """
        if system_prompt is None:
            system_prompt = """你是一个专业的法律法规智能助手。请基于提供的文档内容，准确、详细地回答用户的问题。

要求：
1. 仅基于提供的文档内容回答，不要编造信息
2. 如果文档中没有相关信息，请明确说明
3. 回答要准确、专业、易懂
4. 可以适当引用文档中的具体条款
5. 保持客观中立的态度"""
        
        # 构建上下文
        context_text = self._build_context(context)
        
        # 完整提示词
        prompt = f"""{system_prompt}

{context_text}

=== 用户问题 ===
{question}

=== 回答 ===
请基于上述文档内容回答用户问题："""
        
        return prompt
    
    def generate(
        self, 
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        """
        生成文本回答
        
        Args:
            prompt: 提示词
            temperature: 温度参数
            max_tokens: 最大token数
            **kwargs: 其他参数
            
        Returns:
            生成的文本
        """
        try:
            request_data = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": temperature or self.temperature,
                    "num_predict": max_tokens or self.max_tokens,
                }
            }
            
            # 详细报文打印（受开关控制）
            try:
                if getattr(self.config.logging, 'enable_llm_verbose', False):
                    logger.info("LLM 请求报文:\n" + self._safe_dump(request_data))
                    logger.info(self._pretty_text(prompt, "LLM Prompt", kind='prompt'))
            except Exception:
                pass
            
            logger.debug(f"发送LLM请求: {len(prompt)} 字符")
            
            response = self.client.post(
                f"{self.api_url}/api/generate",
                json=request_data,
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                generated_text = result.get('response', '')
                
                # 打印响应（受开关控制）
                try:
                    if getattr(self.config.logging, 'enable_llm_verbose', False):
                        logger.info("LLM 响应报文:\n" + self._safe_dump(result))
                        logger.info(self._pretty_text(generated_text, "LLM Response", kind='response'))
                except Exception:
                    pass
                
                logger.info(f"LLM生成完成: {len(generated_text)} 字符")
                return generated_text
            else:
                error_msg = f"LLM API调用失败: {response.status_code} - {response.text}"
                logger.error(error_msg)
                return f"抱歉，生成回答时出现错误: {response.status_code}"
                
        except Exception as e:
            logger.error(f"LLM生成失败: {e}")
            return f"抱歉，生成回答时出现错误: {str(e)}"
    
    def generate_stream(
        self, 
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> Generator[str, None, None]:
        """
        流式生成文本
        
        Args:
            prompt: 提示词
            temperature: 温度参数
            max_tokens: 最大token数
            **kwargs: 其他参数
            
        Yields:
            生成的文本片段
        """
        try:
            request_data = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": True,
                "options": {
                    "temperature": temperature or self.temperature,
                    "num_predict": max_tokens or self.max_tokens,
                }
            }
            
            try:
                if getattr(self.config.logging, 'enable_llm_verbose', False):
                    logger.info("LLM 流式请求报文:\n" + self._safe_dump(request_data))
            except Exception:
                pass
            
            logger.debug(f"发送流式LLM请求: {len(prompt)} 字符")
            
            with self.client.stream(
                "POST",
                f"{self.api_url}/api/generate",
                json=request_data,
                timeout=self.timeout
            ) as resp:
                if resp.status_code != 200:
                    logger.error(f"LLM 流式API调用失败: {resp.status_code}")
                    return
                collected = []
                for chunk in resp.iter_lines():
                    if not chunk:
                        continue
                    text = chunk.decode('utf-8', errors='ignore')
                    collected.append(text)
                    yield text
                try:
                    if getattr(self.config.logging, 'enable_llm_verbose', False):
                        sample = "\n".join(collected[:10])
                        max_len = getattr(self.config.logging, 'llm_max_log_chars', 4000)
                        if len(sample) > max_len:
                            sample = sample[:max_len] + f"\n<... truncated {len(sample)-max_len} chars>"
                        logger.info("LLM 流式响应(拼接前)样本:\n" + sample)
                except Exception:
                    pass
                
                logger.info("流式LLM生成完成")
                
        except Exception as e:
            logger.error(f"流式LLM生成失败: {e}")
            yield f"抱歉，生成回答时出现错误: {str(e)}"
    
    async def generate_async(
        self, 
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        """
        异步生成文本
        
        Args:
            prompt: 提示词
            temperature: 温度参数
            max_tokens: 最大token数
            **kwargs: 其他参数
            
        Returns:
            生成的文本
        """
        try:
            request_data = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": temperature or self.temperature,
                    "num_predict": max_tokens or self.max_tokens,
                }
            }
            
            response = await self.async_client.post(
                f"{self.api_url}/api/generate",
                json=request_data,
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get('response', '')
            else:
                logger.error(f"异步LLM API调用失败: {response.status_code}")
                return f"抱歉，生成回答时出现错误: {response.status_code}"
                
        except Exception as e:
            logger.error(f"异步LLM生成失败: {e}")
            return f"抱歉，生成回答时出现错误: {str(e)}"
    
    async def generate_stream_async(
        self, 
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """
        异步流式生成文本
        
        Args:
            prompt: 提示词
            temperature: 温度参数
            max_tokens: 最大token数
            **kwargs: 其他参数
            
        Yields:
            生成的文本片段
        """
        try:
            request_data = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": True,
                "options": {
                    "temperature": temperature or self.temperature,
                    "num_predict": max_tokens or self.max_tokens,
                }
            }
            
            async with self.async_client.stream(
                "POST",
                f"{self.api_url}/api/generate",
                json=request_data,
                timeout=self.timeout
            ) as response:
                
                if response.status_code != 200:
                    yield f"错误: {response.status_code}"
                    return
                
                async for line in response.aiter_lines():
                    if line:
                        try:
                            data = json.loads(line)
                            if 'response' in data:
                                chunk = data['response']
                                if chunk:
                                    yield chunk
                        except json.JSONDecodeError:
                            continue
                
        except Exception as e:
            logger.error(f"异步流式LLM生成失败: {e}")
            yield f"抱歉，生成回答时出现错误: {str(e)}"
    
    def chat(
        self,
        question: str,
        context: List[Dict[str, Any]] = None,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        基于上下文的问答
        
        Args:
            question: 用户问题
            context: 上下文文档列表
            system_prompt: 系统提示词
            **kwargs: 其他参数
            
        Returns:
            包含回答和元信息的字典
        """
        start_time = datetime.now()
        
        # 构建提示词
        prompt = self._build_prompt(question, context or [], system_prompt)
        
        # 生成回答
        answer = self.generate(prompt, **kwargs)
        
        end_time = datetime.now()
        response_time = (end_time - start_time).total_seconds()
        
        return {
            'answer': answer,
            'question': question,
            'response_time': response_time,
            'model': self.model_name,
            'context_count': len(context) if context else 0,
            'timestamp': start_time.isoformat()
        }
    
    def chat_stream(
        self,
        question: str,
        context: List[Dict[str, Any]] = None,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> Generator[str, None, None]:
        """
        基于上下文的流式问答
        
        Args:
            question: 用户问题
            context: 上下文文档列表
            system_prompt: 系统提示词
            **kwargs: 其他参数
            
        Yields:
            回答的文本片段
        """
        # 构建提示词
        prompt = self._build_prompt(question, context or [], system_prompt)
        
        # 流式生成回答
        for chunk in self.generate_stream(prompt, **kwargs):
            yield chunk
    
    async def chat_async(
        self,
        question: str,
        context: List[Dict[str, Any]] = None,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        异步基于上下文的问答
        
        Args:
            question: 用户问题
            context: 上下文文档列表
            system_prompt: 系统提示词
            **kwargs: 其他参数
            
        Returns:
            包含回答和元信息的字典
        """
        start_time = datetime.now()
        
        # 构建提示词
        prompt = self._build_prompt(question, context or [], system_prompt)
        
        # 异步生成回答
        answer = await self.generate_async(prompt, **kwargs)
        
        end_time = datetime.now()
        response_time = (end_time - start_time).total_seconds()
        
        return {
            'answer': answer,
            'question': question,
            'response_time': response_time,
            'model': self.model_name,
            'context_count': len(context) if context else 0,
            'timestamp': start_time.isoformat()
        }
    
    async def chat_stream_async(
        self,
        question: str,
        context: List[Dict[str, Any]] = None,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """
        异步基于上下文的流式问答
        
        Args:
            question: 用户问题
            context: 上下文文档列表
            system_prompt: 系统提示词
            **kwargs: 其他参数
            
        Yields:
            回答的文本片段
        """
        # 构建提示词
        prompt = self._build_prompt(question, context or [], system_prompt)
        
        # 异步流式生成回答
        async for chunk in self.generate_stream_async(prompt, **kwargs):
            yield chunk
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        获取模型信息
        
        Returns:
            模型信息字典
        """
        try:
            response = self.client.post(
                f"{self.api_url}/api/show",
                json={"name": self.model_name}
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"获取模型信息失败: {response.status_code}")
                return {}
                
        except Exception as e:
            logger.error(f"获取模型信息失败: {e}")
            return {}
    
    def close(self) -> None:
        """关闭客户端连接"""
        try:
            self.client.close()
            asyncio.run(self.async_client.aclose())
            logger.info("LLM客户端已关闭")
        except Exception as e:
            logger.error(f"关闭LLM客户端失败: {e}")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close() 

    def _safe_dump(self, data: dict) -> str:
        """将字典安全序列化为可打印字符串，屏蔽大字段并截断长度"""
        try:
            import copy, json as _json
            redacted = copy.deepcopy(data)
            # 可选：隐藏文本型字段，避免与pretty重复
            if getattr(self.config.logging, 'llm_hide_text_in_structured', True):
                if isinstance(redacted, dict):
                    if 'prompt' in redacted:
                        redacted['prompt'] = '<hidden; see LLM Prompt>'
                    if 'response' in redacted:
                        redacted['response'] = '<hidden; see LLM Response>'
            redact_keys = set(getattr(self.config.logging, 'llm_redact_keys', []) or [])
            # 递归屏蔽
            def _walk(obj):
                if isinstance(obj, dict):
                    for k in list(obj.keys()):
                        if k in redact_keys:
                            obj[k] = "<redacted>"
                        else:
                            obj[k] = _walk(obj[k])
                elif isinstance(obj, list):
                    # 对超大列表，仅保留前N项提示
                    if len(obj) > 10:
                        return obj[:3] + ["...", f"<{len(obj)-6} items omitted>", "..."] + obj[-3:]
                return obj
            redacted = _walk(redacted)
            s = _json.dumps(redacted, ensure_ascii=False, indent=2)
            max_len = getattr(self.config.logging, 'llm_max_log_chars', 4000)
            if len(s) > max_len:
                s = s[:max_len] + f"\n<... truncated {len(s)-max_len} chars>"
            return s
        except Exception as e:
            return f"<failed to dump: {e}>"

    def _pretty_text(self, text: str, title: str, kind: str = "") -> str:
        """将长文本按配置截断并美化显示，支持彩色输出(kind: prompt|response)。"""
        try:
            max_len = getattr(self.config.logging, 'llm_max_log_chars', 4000)
            s = text or ""
            if len(s) > max_len:
                s = s[:max_len] + f"\n<... truncated {len(text)-max_len} chars>"
            enable_color = getattr(self.config.logging, 'enable_color', True)
            if enable_color and kind:
                if kind == 'prompt':
                    c = getattr(self.config.logging, 'color_prompt', "\033[36m")
                else:
                    c = getattr(self.config.logging, 'color_response', "\033[32m")
                r = getattr(self.config.logging, 'color_reset', "\033[0m")
                return f"\n{c}===== {title} (len={len(text)}) =====\n{s}\n===== /{title} ====={r}"
            return f"\n===== {title} (len={len(text)}) =====\n{s}\n===== /{title} ====="
        except Exception as e:
            return f"<failed to pretty print {title}: {e}>" 
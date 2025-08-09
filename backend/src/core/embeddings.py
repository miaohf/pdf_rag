"""
嵌入模型

负责文本向量化和嵌入模型管理。
"""

import httpx
import numpy as np
from typing import List, Dict, Any, Optional, Union
import asyncio
from concurrent.futures import ThreadPoolExecutor
import json

from src.utils.config import Config
from src.utils.logger import get_logger

logger = get_logger(__name__)


class EmbeddingModel:
    """嵌入模型"""
    
    def __init__(self, config: Config):
        """
        初始化嵌入模型
        
        Args:
            config: 配置对象
        """
        self.config = config
        self.embedding_config = config.embedding
        self.api_url = self.embedding_config.api_url
        self.model_name = self.embedding_config.model
        self.batch_size = self.embedding_config.batch_size
        self.max_length = self.embedding_config.max_length
        
        # HTTP客户端配置
        self.client = httpx.Client(timeout=30.0)
        self.async_client = httpx.AsyncClient(timeout=30.0)
        
        # 缓存
        self._embedding_cache = {}
        self._model_info = None
        
        logger.info(f"嵌入模型初始化完成: {self.model_name}")
        
        # 验证模型可用性
        self._validate_model()
    
    def _validate_model(self) -> None:
        """验证嵌入模型是否可用"""
        try:
            # 测试连接
            response = self.client.get(f"{self.api_url}/api/tags")
            if response.status_code == 200:
                models = response.json().get('models', [])
                model_names = [m['name'] for m in models]
                
                if self.model_name not in model_names:
                    logger.warning(f"嵌入模型 {self.model_name} 未找到，可用模型: {model_names}")
                else:
                    logger.info(f"嵌入模型 {self.model_name} 验证成功")
            else:
                logger.warning(f"无法连接到Ollama服务: {response.status_code}")
                
        except Exception as e:
            logger.error(f"验证嵌入模型失败: {e}")
    
    def _clean_text(self, text: str) -> str:
        """
        清理和预处理文本
        
        Args:
            text: 原始文本
            
        Returns:
            清理后的文本
        """
        if not text:
            return ""
        
        # 移除多余空白
        text = ' '.join(text.split())
        
        # 截断过长文本
        if len(text) > self.max_length:
            text = text[:self.max_length]
            logger.debug(f"文本被截断到 {self.max_length} 字符")
        
        return text
    
    def embed_text(self, text: str, use_cache: bool = True) -> List[float]:
        """
        对单个文本进行向量化
        
        Args:
            text: 要向量化的文本
            use_cache: 是否使用缓存
            
        Returns:
            文本的向量表示
        """
        text = self._clean_text(text)
        
        if not text:
            logger.warning("空文本，返回零向量")
            return [0.0] * 384  # 默认维度
        
        # 检查缓存
        if use_cache and text in self._embedding_cache:
            logger.debug("使用缓存的嵌入向量")
            return self._embedding_cache[text]
        
        try:
            # 调用Ollama API
            response = self.client.post(
                f"{self.api_url}/api/embeddings",
                json={
                    "model": self.model_name,
                    "prompt": text
                },
                timeout=30.0
            )
            
            if response.status_code == 200:
                embedding = response.json().get('embedding', [])
                
                if embedding:
                    # 缓存结果
                    if use_cache:
                        self._embedding_cache[text] = embedding
                    
                    logger.debug(f"成功生成 {len(embedding)} 维向量")
                    return embedding
                else:
                    logger.error("API返回空向量")
                    return [0.0] * 384
            else:
                logger.error(f"嵌入API调用失败: {response.status_code} - {response.text}")
                return [0.0] * 384
                
        except Exception as e:
            logger.error(f"文本向量化失败: {e}")
            return [0.0] * 384
    
    def embed_batch(self, texts: List[str], use_cache: bool = True) -> List[List[float]]:
        """
        批量向量化文本
        
        Args:
            texts: 文本列表
            use_cache: 是否使用缓存
            
        Returns:
            向量列表
        """
        if not texts:
            return []
        
        logger.info(f"开始批量向量化 {len(texts)} 个文本")
        
        embeddings = []
        
        # 分批处理
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i + self.batch_size]
            batch_embeddings = []
            
            logger.debug(f"处理批次 {i//self.batch_size + 1}: {len(batch_texts)} 个文本")
            
            # 使用线程池并行处理批次内的文本
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = [
                    executor.submit(self.embed_text, text, use_cache)
                    for text in batch_texts
                ]
                
                for future in futures:
                    try:
                        embedding = future.result(timeout=60)
                        batch_embeddings.append(embedding)
                    except Exception as e:
                        logger.error(f"批次向量化失败: {e}")
                        batch_embeddings.append([0.0] * 384)
            
            embeddings.extend(batch_embeddings)
            logger.debug(f"批次 {i//self.batch_size + 1} 完成")
        
        logger.info(f"批量向量化完成: {len(embeddings)} 个向量")
        return embeddings
    
    async def embed_text_async(self, text: str, use_cache: bool = True) -> List[float]:
        """
        异步向量化单个文本
        
        Args:
            text: 要向量化的文本
            use_cache: 是否使用缓存
            
        Returns:
            文本的向量表示
        """
        text = self._clean_text(text)
        
        if not text:
            return [0.0] * 384
        
        # 检查缓存
        if use_cache and text in self._embedding_cache:
            return self._embedding_cache[text]
        
        try:
            response = await self.async_client.post(
                f"{self.api_url}/api/embeddings",
                json={
                    "model": self.model_name,
                    "prompt": text
                },
                timeout=30.0
            )
            
            if response.status_code == 200:
                embedding = response.json().get('embedding', [])
                
                if embedding and use_cache:
                    self._embedding_cache[text] = embedding
                
                return embedding or [0.0] * 384
            else:
                logger.error(f"异步嵌入API调用失败: {response.status_code}")
                return [0.0] * 384
                
        except Exception as e:
            logger.error(f"异步文本向量化失败: {e}")
            return [0.0] * 384
    
    async def embed_batch_async(self, texts: List[str], use_cache: bool = True) -> List[List[float]]:
        """
        异步批量向量化文本
        
        Args:
            texts: 文本列表
            use_cache: 是否使用缓存
            
        Returns:
            向量列表
        """
        if not texts:
            return []
        
        logger.info(f"开始异步批量向量化 {len(texts)} 个文本")
        
        # 分批处理
        embeddings = []
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i + self.batch_size]
            
            # 异步并发处理批次内的文本
            tasks = [
                self.embed_text_async(text, use_cache)
                for text in batch_texts
            ]
            
            batch_embeddings = await asyncio.gather(*tasks, return_exceptions=True)
            
            # 处理异常
            for j, result in enumerate(batch_embeddings):
                if isinstance(result, Exception):
                    logger.error(f"异步向量化失败: {result}")
                    batch_embeddings[j] = [0.0] * 384
            
            embeddings.extend(batch_embeddings)
        
        logger.info(f"异步批量向量化完成: {len(embeddings)} 个向量")
        return embeddings
    
    def get_embedding_dimension(self) -> int:
        """
        获取嵌入向量的维度
        
        Returns:
            向量维度
        """
        if self._model_info is None:
            # 用测试文本获取向量维度
            test_embedding = self.embed_text("测试", use_cache=False)
            self._model_info = {"dimension": len(test_embedding)}
        
        return self._model_info.get("dimension", 384)
    
    def calculate_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """
        计算两个向量的余弦相似度
        
        Args:
            embedding1: 向量1
            embedding2: 向量2
            
        Returns:
            余弦相似度 (0-1)
        """
        try:
            vec1 = np.array(embedding1)
            vec2 = np.array(embedding2)
            
            # 计算余弦相似度
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            similarity = dot_product / (norm1 * norm2)
            
            # 确保结果在 [0, 1] 范围内
            return max(0.0, min(1.0, (similarity + 1) / 2))
            
        except Exception as e:
            logger.error(f"计算相似度失败: {e}")
            return 0.0
    
    def clear_cache(self) -> None:
        """清理缓存"""
        self._embedding_cache.clear()
        logger.info("嵌入向量缓存已清理")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        获取缓存统计信息
        
        Returns:
            缓存统计
        """
        return {
            "cached_embeddings": len(self._embedding_cache),
            "cache_size_mb": sum(len(str(v)) for v in self._embedding_cache.values()) / 1024 / 1024
        }
    
    def close(self) -> None:
        """关闭客户端连接"""
        try:
            self.client.close()
            asyncio.run(self.async_client.aclose())
            logger.info("嵌入模型客户端已关闭")
        except Exception as e:
            logger.error(f"关闭嵌入模型客户端失败: {e}")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close() 
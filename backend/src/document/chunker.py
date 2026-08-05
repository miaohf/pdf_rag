"""
文档分片器

将长文档分割为适合向量化和检索的小片段。
支持基于语义的智能分片和父子分片架构。
"""

import re
import numpy as np
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass

from src.utils.config import Config
from src.utils.logger import get_logger
from src.utils.helpers import clean_text, clean_text_for_technical_docs

# LangChain 语义分片器
try:
    from langchain_text_splitters import (
        RecursiveCharacterTextSplitter,
        SpacyTextSplitter,
        TokenTextSplitter,
        CharacterTextSplitter
    )
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    logger.warning("LangChain text splitters not available, falling back to basic chunking")

logger = get_logger(__name__)


@dataclass
class Chunk:
    """文档片段"""
    content: str  # 分片内容（保留原始格式）
    index: int
    start_pos: int
    end_pos: int
    metadata: Dict[str, Any]
    parent_chunk_id: Optional[str] = None


class SemanticChunker:
    """基于语义的分片器"""
    
    def __init__(self, config: Config):
        """
        初始化语义分片器
        
        Args:
            config: 配置对象
        """
        self.config = config
        self.chunk_size = config.document.chunk_size
        self.chunk_overlap = config.document.chunk_overlap
        
        # 初始化 LangChain 分片器
        self._init_langchain_splitters()
        
        # 语义边界检测配置
        self.semantic_threshold = 0.75  # 语义相似度阈值
        self.min_chunk_size = max(50, self.chunk_size // 4)  # 最小分片大小
        self.max_chunk_size = self.chunk_size * 2  # 最大分片大小
        
        logger.info(f"语义分片器初始化完成: chunk_size={self.chunk_size}, threshold={self.semantic_threshold}")
    
    def _init_langchain_splitters(self):
        """初始化 LangChain 分片器"""
        if not LANGCHAIN_AVAILABLE:
            self.recursive_splitter = None
            self.spacy_splitter = None
            self.token_splitter = None
            return
            
        try:
            # 递归字符分片器 - 优先使用
            self.recursive_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                length_function=len,
                separators=[
                    "\n\n",      # 段落分隔符
                    "\n",        # 行分隔符
                    "。",        # 中文句号
                    "！",        # 中文感叹号
                    "？",        # 中文问号
                    ".",         # 英文句号
                    "!",         # 英文感叹号
                    "?",         # 英文问号
                    ";",         # 分号
                    "；",        # 中文分号
                    ",",         # 逗号
                    "，",        # 中文逗号
                    " ",         # 空格
                    ""           # 字符级别分割
                ]
            )
            
            # Token 分片器 - 精确控制
            self.token_splitter = TokenTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap
            )
            
            # 字符分片器 - 简单分割
            self.char_splitter = CharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                separator="\n"
            )
            
            logger.info("LangChain 分片器初始化成功")
            
        except Exception as e:
            logger.warning(f"LangChain 分片器初始化失败: {e}")
            self.recursive_splitter = None
            self.token_splitter = None
            self.char_splitter = None
    
    def semantic_chunk(
        self, 
        text: str, 
        strategy: str = "recursive",
        preserve_structure: bool = True
    ) -> List[str]:
        """
        基于语义的分片
        
        Args:
            text: 文本内容
            strategy: 分片策略 ("recursive", "token", "character", "auto")
            preserve_structure: 是否保持文档结构
            
        Returns:
            分片文本列表
        """
        if not text.strip():
            return []
        
        # 如果没有 LangChain，降级到基础分片
        if not LANGCHAIN_AVAILABLE:
            return self._basic_semantic_chunk(text)
        
        # 自动选择最佳策略
        if strategy == "auto":
            strategy = self._select_best_strategy(text)
            logger.debug(f"自动选择分片策略: {strategy}")
        
        try:
            # 预处理文本
            processed_text = self._preprocess_text(text, preserve_structure)
            
            # 根据策略选择分片器
            if strategy == "recursive" and self.recursive_splitter:
                chunks = self.recursive_splitter.split_text(processed_text)
            elif strategy == "token" and self.token_splitter:
                chunks = self.token_splitter.split_text(processed_text)
            elif strategy == "character" and self.char_splitter:
                chunks = self.char_splitter.split_text(processed_text)
            else:
                # 降级到递归分片器
                if self.recursive_splitter:
                    chunks = self.recursive_splitter.split_text(processed_text)
                else:
                    chunks = self._basic_semantic_chunk(processed_text)
            
            # 后处理：优化分片质量
            optimized_chunks = self._optimize_chunks(chunks, text)
            
            logger.debug(f"语义分片完成: {len(optimized_chunks)} 个片段 (策略: {strategy})")
            return optimized_chunks
            
        except Exception as e:
            logger.warning(f"语义分片失败，降级到基础分片: {e}")
            return self._basic_semantic_chunk(text)
    
    def _select_best_strategy(self, text: str) -> str:
        """自动选择最佳分片策略"""
        # 分析文本特征
        line_count = len(text.split('\n'))
        paragraph_count = len(re.split(r'\n\s*\n', text))
        sentence_count = len(re.split(r'[。！？.!?]+', text))
        
        # 检查技术文档特征
        numbered_items = len(re.findall(r'\n\s*\d+\.\d+', text))
        technical_patterns = len(re.findall(r'(km/h|metres?|Category|Regulation|UNECE)', text))
        
        # 策略选择逻辑
        if numbered_items > 3 or technical_patterns > 5:
            # 技术规格文档：使用递归分片保持结构
            return "recursive"
        elif len(text) > 10000:
            # 长文档：使用 token 分片确保长度控制
            return "token"
        elif paragraph_count > 5:
            # 段落结构明显：使用递归分片
            return "recursive"
        else:
            # 默认使用递归分片
            return "recursive"
    
    def _preprocess_text(self, text: str, preserve_structure: bool) -> str:
        """预处理文本"""
        if not preserve_structure:
            # 标准化空白字符
            text = re.sub(r'\n\s*\n', '\n\n', text)  # 标准化段落分隔
            text = re.sub(r'[ \t]+', ' ', text)       # 标准化空格和制表符
            
        # 移除过多的空行
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        return text.strip()
    
    def _optimize_chunks(self, chunks: List[str], original_text: str) -> List[str]:
        """优化分片质量"""
        if not chunks:
            return chunks
        
        optimized = []
        
        for chunk in chunks:
            chunk = chunk.strip()
            if not chunk:
                continue
            
            # 检查分片大小
            if len(chunk) < self.min_chunk_size:
                # 太小的分片尝试与前一个合并
                if optimized and len(optimized[-1]) + len(chunk) <= self.max_chunk_size:
                    optimized[-1] = optimized[-1] + '\n' + chunk
                    continue
            
            # 检查是否是不完整的句子
            if not self._is_complete_sentence(chunk):
                # 尝试与前一个分片合并
                if optimized and len(optimized[-1]) + len(chunk) <= self.max_chunk_size:
                    optimized[-1] = optimized[-1] + '\n' + chunk
                    continue
            
            optimized.append(chunk)
        
        return optimized
    
    def _is_complete_sentence(self, text: str) -> bool:
        """检查文本是否是完整的句子"""
        text = text.strip()
        if not text:
            return False
        
        # 检查是否以句号、问号、感叹号结尾
        if text[-1] in '。！？.!?':
            return True
        
        # 检查是否是列表项
        if re.match(r'^\d+\.\s', text) or re.match(r'^[•\-\*]\s', text):
            return True
        
        # 检查是否是标题
        if len(text.split()) <= 10 and not text.endswith(',，;；'):
            return True
        
        return False
    
    def _basic_semantic_chunk(self, text: str) -> List[str]:
        """基础语义分片（不依赖 LangChain）"""
        # 按段落分割
        paragraphs = re.split(r'\n\s*\n', text)
        
        chunks = []
        current_chunk = ""
        
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            
            # 检查添加这个段落是否会超过限制
            test_chunk = current_chunk + '\n\n' + paragraph if current_chunk else paragraph
            
            if len(test_chunk) > self.chunk_size and current_chunk:
                # 保存当前分片
                chunks.append(current_chunk.strip())
                current_chunk = paragraph
            else:
                current_chunk = test_chunk
        
        # 添加最后一个分片
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return chunks


class DocumentChunker:
    """文档分片器 - 支持语义分片和父子分片架构"""
    
    def __init__(self, config: Config):
        """
        初始化分片器
        
        Args:
            config: 配置对象
        """
        self.config = config
        self.chunk_size = config.document.chunk_size
        self.chunk_overlap = config.document.chunk_overlap
        self.parent_chunk_size = getattr(config.document, 'parent_chunk_size', self.chunk_size * 3)
        self.use_hierarchical_chunking = getattr(config.document, 'use_hierarchical_chunking', False)
        self.max_chunks = config.document.max_chunks_per_document
        
        # 初始化语义分片器
        self.semantic_chunker = SemanticChunker(config)
        
        # 分片策略配置
        self.use_semantic_chunking = getattr(config.document, 'use_semantic_chunking', True)
        self.semantic_strategy = getattr(config.document, 'semantic_strategy', 'auto')
        
        logger.info(f"文档分片器初始化完成: "
                   f"子分片={self.chunk_size}, 父分片={self.parent_chunk_size}, "
                   f"重叠={self.chunk_overlap}, 层级分片={self.use_hierarchical_chunking}, "
                   f"语义分片={self.use_semantic_chunking}")
    
    def chunk_by_sentences(self, text: str, original_text: Optional[str] = None) -> List[Chunk]:
        """
        按句子分割文档，优化处理技术文档
        
        Args:
            text: 文本内容
            original_text: 原始文本内容
            
        Returns:
            文档片段列表
        """
        # 如果启用语义分片，使用语义分片器
        if self.use_semantic_chunking and LANGCHAIN_AVAILABLE:
            return self._semantic_chunk_to_chunks(text, original_text, "recursive")
        
        # 原有的句子分片逻辑
        effective_chunk_size = self.chunk_size
        effective_overlap = self.chunk_overlap
        
        if self.use_hierarchical_chunking:
            logger.debug(f"使用层级分片模式，子分片大小: {effective_chunk_size}")
        
        lines = text.split('\n')
        original_lines = (original_text or text).split('\n')
        
        chunks = []
        current_chunk = ""
        current_original_chunk = ""
        current_start = 0
        chunk_index = 0
        
        for i, line in enumerate(lines):
            line = line.strip()
            original_line = original_lines[i].strip() if i < len(original_lines) else line
            
            if not line:
                continue
            
            test_chunk = current_chunk + "\n" + line if current_chunk else line
            test_original_chunk = current_original_chunk + "\n" + original_line if current_original_chunk else original_line
            
            if len(test_chunk) > effective_chunk_size and current_chunk:
                cleaned_chunk_content = clean_text_for_technical_docs(current_chunk.strip())
                
                chunk = Chunk(
                    content=current_original_chunk.strip(),
                    index=chunk_index,
                    start_pos=current_start,
                    end_pos=current_start + len(current_chunk),
                    metadata={
                        'type': 'line_based_optimized',
                        'contains_numbered_items': bool(re.search(r'\d+\.\d+', current_chunk)),
                        'effective_chunk_size': effective_chunk_size,
                        'line_count': len(current_chunk.split('\n')),
                        'chunking_method': 'sentence_based'
                    }
                )
                chunks.append(chunk)
                
                # 处理重叠
                if effective_overlap > 0:
                    lines_in_chunk = current_chunk.split('\n')
                    original_lines_in_chunk = current_original_chunk.split('\n')
                    if len(lines_in_chunk) > 2:
                        overlap_lines = lines_in_chunk[-2:]
                        overlap_original_lines = original_lines_in_chunk[-2:]
                        overlap_text = '\n'.join(overlap_lines)
                        overlap_original_text = '\n'.join(overlap_original_lines)
                        current_chunk = overlap_text + "\n" + line
                        current_original_chunk = overlap_original_text + "\n" + original_line
                        current_start = chunk.end_pos - len(overlap_text)
                    else:
                        current_chunk = line
                        current_original_chunk = original_line
                        current_start = chunk.end_pos
                else:
                    current_chunk = line
                    current_original_chunk = original_line
                    current_start = chunk.end_pos
                
                chunk_index += 1
                
                if chunk_index >= self.max_chunks:
                    logger.warning(f"达到最大片段数限制: {self.max_chunks}")
                    break
            else:
                current_chunk = test_chunk
                current_original_chunk = test_original_chunk
        
        # 添加最后一个片段
        if current_chunk.strip() and chunk_index < self.max_chunks:
            cleaned_chunk_content = clean_text_for_technical_docs(current_chunk.strip())
            
            chunk = Chunk(
                content=current_original_chunk.strip(),
                index=chunk_index,
                start_pos=current_start,
                end_pos=current_start + len(current_chunk),
                metadata={
                    'type': 'line_based_optimized',
                    'contains_numbered_items': bool(re.search(r'\d+\.\d+', current_chunk)),
                    'effective_chunk_size': effective_chunk_size,
                    'line_count': len(current_chunk.split('\n')),
                    'chunking_method': 'sentence_based'
                }
            )
            chunks.append(chunk)
        
        logger.debug(f"句子分片完成: {len(chunks)} 个片段")
        return chunks
    
    def _semantic_chunk_to_chunks(
        self, 
        text: str, 
        original_text: Optional[str], 
        strategy: str = "auto"
    ) -> List[Chunk]:
        """
        将语义分片转换为 Chunk 对象
        
        Args:
            text: 文本内容
            original_text: 原始文本内容
            strategy: 分片策略
            
        Returns:
            Chunk 对象列表
        """
        # 使用语义分片器
        chunk_texts = self.semantic_chunker.semantic_chunk(
            text, 
            strategy=strategy, 
            preserve_structure=True
        )
        
        chunks = []
        current_pos = 0
        original_text = original_text or text
        
        for i, chunk_text in enumerate(chunk_texts):
            # 在原文中找到这个分片的位置
            start_pos = original_text.find(chunk_text.strip()[:50], current_pos)
            if start_pos == -1:
                start_pos = current_pos
            
            end_pos = start_pos + len(chunk_text)
            
            # 确保我们获取的是原始格式的内容
            if start_pos >= 0 and end_pos <= len(original_text):
                original_chunk_content = original_text[start_pos:end_pos]
            else:
                original_chunk_content = chunk_text
            
            chunk = Chunk(
                content=original_chunk_content.strip(),
                index=i,
                start_pos=start_pos,
                end_pos=end_pos,
                metadata={
                    'type': 'semantic_chunk',
                    'chunking_method': 'langchain_semantic',
                    'strategy': strategy,
                    'chunk_size': len(chunk_text),
                    'contains_numbered_items': bool(re.search(r'\d+\.\d+', chunk_text)),
                    'semantic_quality': self._assess_semantic_quality(chunk_text)
                }
            )
            chunks.append(chunk)
            current_pos = end_pos
            
            # 检查最大片段数限制
            if len(chunks) >= self.max_chunks:
                logger.warning(f"达到最大片段数限制: {self.max_chunks}")
                break
        
        logger.debug(f"语义分片转换完成: {len(chunks)} 个片段")
        return chunks
    
    def _assess_semantic_quality(self, text: str) -> float:
        """评估分片的语义质量"""
        if not text.strip():
            return 0.0
        
        quality_score = 0.5  # 基础分数
        
        # 完整性检查
        if text.strip()[-1] in '。！？.!?':
            quality_score += 0.2
        
        # 长度适中性
        length = len(text)
        if self.chunk_size * 0.3 <= length <= self.chunk_size * 1.2:
            quality_score += 0.2
        
        # 结构完整性
        if not text.startswith(' ') and not text.endswith(' '):
            quality_score += 0.1
        
        return min(1.0, quality_score)
    
    def chunk_by_paragraphs(self, text: str, original_text: Optional[str] = None) -> List[Chunk]:
        """
        按段落分割文档
        
        Args:
            text: 文本内容
            original_text: 原始文本内容
            
        Returns:
            文档片段列表
        """
        # 如果启用语义分片，使用语义分片器
        if self.use_semantic_chunking and LANGCHAIN_AVAILABLE:
            return self._semantic_chunk_to_chunks(text, original_text, "recursive")
        
        # 原有的段落分片逻辑保持不变
        # ... (保留原有实现)
        return self._legacy_paragraph_chunk(text, original_text)
    
    def _legacy_paragraph_chunk(self, text: str, original_text: Optional[str] = None) -> List[Chunk]:
        """传统的段落分片方法"""
        paragraphs = re.split(r'\n\s*\n', text)
        original_paragraphs = re.split(r'\n\s*\n', original_text or text)
        
        chunks = []
        current_chunk = ""
        current_original_chunk = ""
        current_start = 0
        chunk_index = 0
        
        for i, paragraph in enumerate(paragraphs):
            paragraph = paragraph.strip()
            original_paragraph = original_paragraphs[i].strip() if i < len(original_paragraphs) else paragraph
            
            if not paragraph:
                continue
            
            test_chunk = current_chunk + "\n\n" + paragraph if current_chunk else paragraph
            test_original_chunk = current_original_chunk + "\n\n" + original_paragraph if current_original_chunk else original_paragraph
            
            if len(test_chunk) > self.chunk_size and current_chunk:
                chunk = Chunk(
                    content=current_original_chunk.strip(),
                    index=chunk_index,
                    start_pos=current_start,
                    end_pos=current_start + len(current_chunk),
                    metadata={
                        'type': 'paragraph_based',
                        'paragraph_count': len(current_chunk.split('\n\n')),
                        'chunking_method': 'paragraph_based'
                    }
                )
                chunks.append(chunk)
                
                # 处理重叠
                if self.chunk_overlap > 0:
                    overlap_text = current_chunk[-self.chunk_overlap:]
                    overlap_original_text = current_original_chunk[-self.chunk_overlap:]
                    current_chunk = overlap_text + "\n\n" + paragraph
                    current_original_chunk = overlap_original_text + "\n\n" + original_paragraph
                    current_start = chunk.end_pos - len(overlap_text)
                else:
                    current_chunk = paragraph
                    current_original_chunk = original_paragraph
                    current_start = chunk.end_pos
                
                chunk_index += 1
                
                if chunk_index >= self.max_chunks:
                    break
            else:
                current_chunk = test_chunk
                current_original_chunk = test_original_chunk
        
        # 添加最后一个片段
        if current_chunk.strip() and chunk_index < self.max_chunks:
            chunk = Chunk(
                content=current_original_chunk.strip(),
                index=chunk_index,
                start_pos=current_start,
                end_pos=current_start + len(current_chunk),
                metadata={
                    'type': 'paragraph_based',
                    'paragraph_count': len(current_chunk.split('\n\n')),
                    'chunking_method': 'paragraph_based'
                }
            )
            chunks.append(chunk)
        
        return chunks
    
    def chunk_by_fixed_size(self, text: str, original_text: Optional[str] = None) -> List[Chunk]:
        """
        按固定大小分割文档
        
        Args:
            text: 文本内容
            original_text: 原始文本内容
            
        Returns:
            文档片段列表
        """
        # 如果启用语义分片，使用 token 分片器确保精确控制长度
        if self.use_semantic_chunking and LANGCHAIN_AVAILABLE:
            return self._semantic_chunk_to_chunks(text, original_text, "token")
        
        # 原有的固定大小分片逻辑保持不变
        # ... (保留原有实现)
        return self._legacy_fixed_size_chunk(text, original_text)
    
    def _legacy_fixed_size_chunk(self, text: str, original_text: Optional[str] = None) -> List[Chunk]:
        """传统的固定大小分片方法"""
        chunks = []
        original_text = original_text or text
        
        start_pos = 0
        chunk_index = 0
        
        while start_pos < len(text) and chunk_index < self.max_chunks:
            end_pos = min(start_pos + self.chunk_size, len(text))
            
            # 尝试在合适的位置断开
            if end_pos < len(text):
                for i in range(end_pos, max(start_pos + self.chunk_size // 2, end_pos - 100), -1):
                    if text[i] in '\n。！？.!?':
                        end_pos = i + 1
                        break
            
            chunk_text = text[start_pos:end_pos].strip()
            original_chunk_text = original_text[start_pos:end_pos].strip()
            
            if chunk_text:
                chunk = Chunk(
                    content=original_chunk_text,
                    index=chunk_index,
                    start_pos=start_pos,
                    end_pos=end_pos,
                    metadata={
                        'type': 'fixed_size',
                        'actual_size': len(chunk_text),
                        'chunking_method': 'fixed_size'
                    }
                )
                chunks.append(chunk)
                chunk_index += 1
            
            start_pos = max(start_pos + self.chunk_size - self.chunk_overlap, end_pos)
        
        return chunks
    
    def smart_chunk(self, text: str, strategy: str = "auto") -> List[Chunk]:
        """
        智能分片，根据文档类型选择最佳策略
        
        Args:
            text: 文本内容
            strategy: 分片策略 ("auto", "sentences", "paragraphs", "fixed", "semantic")
            
        Returns:
            文档片段列表
        """
        original_text = text
        
        if not text.strip():
            return []
        
        # 如果明确指定语义分片
        if strategy == "semantic" and self.use_semantic_chunking:
            return self._semantic_chunk_to_chunks(text, original_text, "auto")
        
        # 自动选择策略
        if strategy == "auto":
            # 检查段落结构
            paragraph_count = len(re.split(r'\n\s*\n', text))
            sentence_count = len(re.split(r'[。！？.!?]+', text))
            
            # 检查是否是技术规格文档
            numbered_items = len(re.findall(r'\n\s*\d+\.\d+\s*[A-Z]', text))
            technical_patterns = len(re.findall(r'(km/h|metres?|Category|Regulation|UNECE)', text))
            
            # 策略选择逻辑
            if self.use_semantic_chunking and LANGCHAIN_AVAILABLE:
                # 优先使用语义分片
                if numbered_items > 3 or technical_patterns > 5:
                    logger.info(f"检测到技术规格文档，使用语义分片保持结构完整性")
                    return self._semantic_chunk_to_chunks(text, original_text, "recursive")
                elif len(text) > 5000:
                    logger.info(f"检测到长文档，使用语义分片优化语义连贯性")
                    return self._semantic_chunk_to_chunks(text, original_text, "auto")
                else:
                    logger.info(f"使用语义分片处理常规文档")
                    return self._semantic_chunk_to_chunks(text, original_text, "recursive")
            else:
                # 降级到传统策略
                if numbered_items > 3 or technical_patterns > 5:
                    strategy = "sentences"
                    logger.info(f"检测到技术规格文档，使用句子分片")
                elif paragraph_count > 5 and len(text) / paragraph_count > 200:
                    strategy = "paragraphs"
                elif sentence_count > 10:
                    strategy = "sentences"
                else:
                    strategy = "fixed"
                
                logger.info(f"自动选择分片策略: {strategy}")
        
        # 执行分片
        if strategy == "sentences":
            chunks = self.chunk_by_sentences(text, original_text)
        elif strategy == "paragraphs":
            chunks = self.chunk_by_paragraphs(text, original_text)
        elif strategy == "fixed":
            chunks = self.chunk_by_fixed_size(text, original_text)
        else:
            raise ValueError(f"不支持的分片策略: {strategy}")
        
        logger.info(f"文档分片完成: 生成 {len(chunks)} 个片段")
        return chunks
    
    def create_parent_child_chunks(
        self, 
        text: str, 
        original_text: Optional[str] = None, 
        parent_size: Optional[int] = None
    ) -> Dict[str, List[Chunk]]:
        """
        创建父子文档片段 - 父分片使用传统方法，子分片使用语义分片
        
        Args:
            text: 文本内容
            original_text: 原始文本内容
            parent_size: 父片段大小
            
        Returns:
            包含父片段和子片段的字典
        """
        if parent_size is None:
            parent_size = self.parent_chunk_size
        
        logger.info(f"开始创建父子分片: 父分片大小={parent_size}, 子分片大小={self.chunk_size}")
        
        # 创建父片段（使用传统智能分片，确保稳定性）
        old_chunk_size = self.chunk_size
        old_overlap = self.chunk_overlap
        old_semantic = self.use_semantic_chunking
        
        # 临时调整为父分片参数，禁用语义分片确保大块稳定
        self.chunk_size = parent_size
        self.chunk_overlap = int(parent_size * 0.1)
        self.use_semantic_chunking = False  # 父分片使用传统方法
        
        try:
            parent_chunks = self.smart_chunk(text, strategy="auto")
        finally:
            # 恢复原始参数
            self.chunk_size = old_chunk_size
            self.chunk_overlap = old_overlap
            self.use_semantic_chunking = old_semantic
        
        logger.info(f"父分片创建完成: {len(parent_chunks)} 个父分片")
        
        # 为每个父片段创建子片段（使用语义分片）
        child_chunks = []
        
        for parent_idx, parent_chunk in enumerate(parent_chunks):
            # 为父分片添加层级标识
            parent_chunk.metadata.update({
                'chunk_level': 'parent',
                'hierarchy_level': 2,
                'child_count': 0,
                'chunk_id': f"parent_{parent_idx}",
                'semantic_chunking_enabled': False
            })
            
            # 从父分片中创建子分片（使用语义分片）
            if self.use_semantic_chunking and LANGCHAIN_AVAILABLE:
                logger.debug(f"为父分片 {parent_idx} 创建语义子分片")
                children = self._semantic_chunk_to_chunks(
                    parent_chunk.content, 
                    parent_chunk.content, 
                    strategy="auto"
                )
            else:
                logger.debug(f"为父分片 {parent_idx} 创建传统子分片")
                children = self.smart_chunk(parent_chunk.content, strategy="auto")
            
            # 设置父子关系和元数据
            for child_idx, child in enumerate(children):
                child.parent_chunk_id = f"parent_{parent_idx}"
                child.index = len(child_chunks)  # 全局重新编号
                child.metadata.update({
                    'chunk_level': 'child',
                    'hierarchy_level': 1,
                    'parent_index': parent_idx,
                    'child_sequence': child_idx,
                    'chunk_id': f"child_{len(child_chunks)}",
                    'semantic_chunking_enabled': self.use_semantic_chunking and LANGCHAIN_AVAILABLE
                })
                child_chunks.append(child)
            
            # 更新父分片的子分片数量
            parent_chunk.metadata['child_count'] = len(children)
            
            logger.debug(f"父分片 {parent_idx} 产生 {len(children)} 个子分片")
        
        logger.info(f"父子分片创建完成: {len(parent_chunks)} 个父分片, {len(child_chunks)} 个子分片")
        
        return {
            'parent_chunks': parent_chunks,
            'child_chunks': child_chunks
        } 
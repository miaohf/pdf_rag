"""
文档分片器

将长文档分割为适合向量化和检索的小片段。
"""

import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from src.utils.config import Config
from src.utils.logger import get_logger
from src.utils.helpers import clean_text, clean_text_for_technical_docs

logger = get_logger(__name__)


@dataclass
class Chunk:
    """文档片段"""
    content: str  # 纯文本内容
    index: int
    start_pos: int
    end_pos: int
    metadata: Dict[str, Any]
    original_content: Optional[str] = None  # 保留格式的原始内容，可选
    parent_chunk_id: Optional[str] = None
    
    def __post_init__(self):
        """后处理：如果original_content为空，则设为content"""
        if self.original_content is None:
            self.original_content = self.content


class DocumentChunker:
    """文档分片器"""
    
    def __init__(self, config: Config):
        """
        初始化分片器
        
        Args:
            config: 配置对象
        """
        self.config = config
        self.chunk_size = config.document.chunk_size
        self.chunk_overlap = config.document.chunk_overlap
        self.max_chunks = config.document.max_chunks_per_document
        
        logger.info(f"初始化文档分片器: 片段大小={self.chunk_size}, 重叠={self.chunk_overlap}")
    
    def chunk_by_sentences(self, text: str, original_text: Optional[str] = None) -> List[Chunk]:
        """
        按句子分割文档，优化处理技术文档
        
        Args:
            text: 文本内容
            
        Returns:
            文档片段列表
        """
        # 使用配置的片段大小，对技术文档稍作调整
        effective_chunk_size = self.chunk_size
        effective_overlap = self.chunk_overlap
        
        # 简化的分片策略：按行处理，保持编号项目完整
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
            
            # 检查添加这一行是否会超过限制
            test_chunk = current_chunk + "\n" + line if current_chunk else line
            test_original_chunk = current_original_chunk + "\n" + original_line if current_original_chunk else original_line
            
            if len(test_chunk) > effective_chunk_size and current_chunk:
                # 清理current_chunk用于向量匹配（使用技术文档专用清理）
                cleaned_chunk_content = clean_text_for_technical_docs(current_chunk.strip())
                
                # 保存当前片段
                chunk = Chunk(
                    content=cleaned_chunk_content,  # 清理后的文本用于向量匹配
                    original_content=current_original_chunk.strip(),  # 原始格式用于显示
                    index=chunk_index,
                    start_pos=current_start,
                    end_pos=current_start + len(current_chunk),
                    metadata={
                        'type': 'line_based_optimized',
                        'contains_numbered_items': bool(re.search(r'\d+\.\d+', current_chunk)),
                        'effective_chunk_size': effective_chunk_size,
                        'line_count': len(current_chunk.split('\n'))
                    }
                )
                chunks.append(chunk)
                
                # 处理重叠 - 保留最后几行作为上下文
                if effective_overlap > 0:
                    lines_in_chunk = current_chunk.split('\n')
                    original_lines_in_chunk = current_original_chunk.split('\n')
                    if len(lines_in_chunk) > 2:
                        # 保留最后1-2行作为重叠
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
                
                # 检查是否超过最大片段数
                if chunk_index >= self.max_chunks:
                    logger.warning(f"达到最大片段数限制: {self.max_chunks}")
                    break
            else:
                current_chunk = test_chunk
                current_original_chunk = test_original_chunk
        
        # 添加最后一个片段
        if current_chunk.strip() and chunk_index < self.max_chunks:
            # 清理最后一个片段用于向量匹配（使用技术文档专用清理）
            cleaned_chunk_content = clean_text_for_technical_docs(current_chunk.strip())
            
            chunk = Chunk(
                content=cleaned_chunk_content,  # 清理后的文本用于向量匹配
                original_content=current_original_chunk.strip(),  # 原始格式用于显示
                index=chunk_index,
                start_pos=current_start,
                end_pos=current_start + len(current_chunk),
                metadata={
                    'type': 'line_based_optimized',
                    'contains_numbered_items': bool(re.search(r'\d+\.\d+', current_chunk)),
                    'effective_chunk_size': self.chunk_size,
                    'line_count': len(current_chunk.split('\n'))
                }
            )
            chunks.append(chunk)
        
        return chunks
    
    def chunk_by_paragraphs(self, text: str, original_text: Optional[str] = None) -> List[Chunk]:
        """
        按段落分割文档
        
        Args:
            text: 文本内容
            original_text: 原始文本内容（可选）
            
        Returns:
            文档片段列表
        """
        paragraphs = re.split(r'\n\s*\n', text)
        
        chunks = []
        current_chunk = ""
        current_start = 0
        chunk_index = 0
        
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            
            # 检查添加这个段落是否会超过限制
            test_chunk = current_chunk + "\n\n" + paragraph if current_chunk else paragraph
            
            if len(test_chunk) > self.chunk_size and current_chunk:
                # 保存当前片段
                cleaned_content = clean_text_for_technical_docs(current_chunk.strip())
                chunk = Chunk(
                    content=cleaned_content,
                    original_content=current_chunk.strip(),  # 使用未清理的文本作为original_content
                    index=chunk_index,
                    start_pos=current_start,
                    end_pos=current_start + len(current_chunk),
                    metadata={
                        'type': 'paragraph_based',
                        'paragraph_count': len(current_chunk.split('\n\n'))
                    }
                )
                chunks.append(chunk)
                
                # 处理重叠
                if self.chunk_overlap > 0:
                    overlap_text = current_chunk[-self.chunk_overlap:]
                    current_chunk = overlap_text + "\n\n" + paragraph
                    current_start = chunk.end_pos - len(overlap_text)
                else:
                    current_chunk = paragraph
                    current_start = chunk.end_pos
                
                chunk_index += 1
                
                # 检查是否超过最大片段数
                if chunk_index >= self.max_chunks:
                    logger.warning(f"达到最大片段数限制: {self.max_chunks}")
                    break
            else:
                if current_chunk:
                    current_chunk += "\n\n" + paragraph
                else:
                    current_chunk = paragraph
        
        # 添加最后一个片段
        if current_chunk.strip() and chunk_index < self.max_chunks:
            cleaned_content = clean_text_for_technical_docs(current_chunk.strip())
            chunk = Chunk(
                content=cleaned_content,
                original_content=current_chunk.strip(),
                index=chunk_index,
                start_pos=current_start,
                end_pos=current_start + len(current_chunk),
                metadata={
                    'type': 'paragraph_based',
                    'paragraph_count': len(current_chunk.split('\n\n'))
                }
            )
            chunks.append(chunk)
        
        return chunks
    
    def chunk_by_fixed_size(self, text: str, original_text: Optional[str] = None) -> List[Chunk]:
        """
        按固定大小分割文档
        
        Args:
            text: 文本内容
            original_text: 原始文本内容（可选）
            
        Returns:
            文档片段列表
        """
        chunks = []
        text_length = len(text)
        start_pos = 0
        chunk_index = 0
        
        while start_pos < text_length and chunk_index < self.max_chunks:
            end_pos = min(start_pos + self.chunk_size, text_length)
            
            # 如果不是最后一块，尝试在合适的位置分割
            if end_pos < text_length:
                # 向前寻找换行符或空格
                for i in range(end_pos, max(start_pos + self.chunk_size // 2, end_pos - 100), -1):
                    if i < len(text) and text[i] in '\n ':
                            end_pos = i + 1
                            break
            
            chunk_text = text[start_pos:end_pos].strip()
            if chunk_text:
                cleaned_content = clean_text_for_technical_docs(chunk_text)
                chunk = Chunk(
                    content=cleaned_content,
                    original_content=chunk_text,  # 使用未清理的文本作为original_content
                    index=chunk_index,
                    start_pos=start_pos,
                    end_pos=end_pos,
                    metadata={
                        'type': 'fixed_size',
                        'char_count': len(chunk_text)
                    }
                )
                chunks.append(chunk)
                chunk_index += 1
            
            # 处理重叠
            if self.chunk_overlap > 0:
                start_pos = max(start_pos + self.chunk_size - self.chunk_overlap, end_pos)
            else:
                start_pos = end_pos
            
            # 防止无限循环
            if start_pos <= chunks[-1].end_pos if chunks else start_pos >= end_pos:
                start_pos = end_pos
        
        return chunks
    
    def smart_chunk(self, text: str, strategy: str = "auto") -> List[Chunk]:
        """
        智能分片，根据文档类型选择最佳策略
        
        Args:
            text: 文本内容
            strategy: 分片策略 ("auto", "sentences", "paragraphs", "fixed")
            
        Returns:
            文档片段列表
        """
        # 保存原始文本用于格式保留
        original_text = text
        
        if not text.strip():
            return []
        
        # 自动选择策略
        if strategy == "auto":
            # 检查段落结构
            paragraph_count = len(re.split(r'\n\s*\n', text))
            sentence_count = len(re.split(r'[。！？.!?]+', text))
            
            # 检查是否是技术规格文档（包含编号列表）
            numbered_items = len(re.findall(r'\n\s*\d+\.\d+\s*[A-Z]', text))
            technical_patterns = len(re.findall(r'(km/h|metres?|Category|Regulation|UNECE)', text))
            
            # 优化策略选择逻辑
            if numbered_items > 3 or technical_patterns > 5:
                # 技术规格文档，使用句子分片确保完整性
                strategy = "sentences"
                logger.info(f"检测到技术规格文档 (编号项目: {numbered_items}, 技术术语: {technical_patterns})")
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
    
    def create_parent_child_chunks(self, text: str, parent_size: Optional[int] = None) -> Dict[str, List[Chunk]]:
        """
        创建父子文档片段
        
        Args:
            text: 文本内容
            parent_size: 父片段大小，默认为chunk_size的3倍
            
        Returns:
            包含父片段和子片段的字典
        """
        if parent_size is None:
            parent_size = self.chunk_size * 3
        
        # 创建父片段
        old_chunk_size = self.chunk_size
        self.chunk_size = parent_size
        parent_chunks = self.chunk_by_fixed_size(text)
        self.chunk_size = old_chunk_size
        
        # 为每个父片段创建子片段
        child_chunks = []
        
        for parent_chunk in parent_chunks:
            children = self.smart_chunk(parent_chunk.content)
            
            # 设置父子关系
            for child in children:
                child.parent_chunk_id = f"parent_{parent_chunk.index}"
                child.index = len(child_chunks)  # 重新编号
                child_chunks.append(child)
        
        logger.info(f"创建父子片段: {len(parent_chunks)} 个父片段, {len(child_chunks)} 个子片段")
        
        return {
            'parent_chunks': parent_chunks,
            'child_chunks': child_chunks
        } 
"""
文档加载器

支持多种文档格式的加载和文本提取。
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
from abc import ABC, abstractmethod

from src.utils.logger import get_logger
from src.utils.helpers import calculate_file_hash, format_file_size, extract_metadata

logger = get_logger(__name__)


class BaseLoader(ABC):
    """文档加载器基类"""
    
    @abstractmethod
    def load(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        加载文档
        
        Args:
            file_path: 文档路径
            
        Returns:
            包含文档内容和元数据的字典
        """
        pass
    
    @abstractmethod
    def supports(self, file_path: Union[str, Path]) -> bool:
        """
        检查是否支持该文件格式
        
        Args:
            file_path: 文档路径
            
        Returns:
            是否支持
        """
        pass


class PDFLoader(BaseLoader):
    """PDF文档加载器"""
    
    def __init__(self):
        self.supported_extensions = {'.pdf'}
    
    def supports(self, file_path: Union[str, Path]) -> bool:
        """检查是否为PDF文件"""
        return Path(file_path).suffix.lower() in self.supported_extensions
    
    def load(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        加载PDF文档
        
        Args:
            file_path: PDF文件路径
            
        Returns:
            文档内容和元数据
        """
        try:
            import pypdf
            
            file_path = Path(file_path)
            
            # 检查文件是否存在
            if not file_path.exists():
                raise FileNotFoundError(f"文件不存在: {file_path}")
            
            # 获取文件信息
            file_stat = file_path.stat()
            file_hash = calculate_file_hash(file_path)
            
            # 读取PDF内容
            text_content = []
            metadata = {}
            
            with open(file_path, 'rb') as file:
                pdf_reader = pypdf.PdfReader(file)
                
                # 提取元数据
                if pdf_reader.metadata:
                    metadata.update({
                        'title': pdf_reader.metadata.get('/Title', ''),
                        'author': pdf_reader.metadata.get('/Author', ''),
                        'subject': pdf_reader.metadata.get('/Subject', ''),
                        'creator': pdf_reader.metadata.get('/Creator', ''),
                        'producer': pdf_reader.metadata.get('/Producer', ''),
                        'creation_date': str(pdf_reader.metadata.get('/CreationDate', '')),
                        'modification_date': str(pdf_reader.metadata.get('/ModDate', ''))
                    })
                
                # 提取文本内容
                for page_num, page in enumerate(pdf_reader.pages):
                    try:
                        page_text = page.extract_text()
                        if page_text.strip():
                            text_content.append(f"=== 第 {page_num + 1} 页 ===\n{page_text}")
                    except Exception as e:
                        logger.warning(f"提取第 {page_num + 1} 页内容失败: {e}")
                        continue
            
            # 合并所有文本
            full_text = '\n\n'.join(text_content)
            
            # 提取文本元数据
            text_metadata = extract_metadata(full_text)
            metadata.update(text_metadata)
            
            return {
                'content': full_text,
                'metadata': {
                    **metadata,
                    'filename': file_path.name,
                    'file_path': str(file_path.absolute()),
                    'file_size': file_stat.st_size,
                    'file_hash': file_hash,
                    'content_type': 'application/pdf',
                    'page_count': len(pdf_reader.pages),
                    'loader': 'PDFLoader'
                }
            }
            
        except ImportError:
            raise ImportError("请安装pypdf库: pip install pypdf")
        except Exception as e:
            logger.error(f"加载PDF文件失败 {file_path}: {e}")
            raise


class MarkdownLoader(BaseLoader):
    """Markdown文档加载器"""
    
    def __init__(self):
        self.supported_extensions = {'.md', '.markdown'}
    
    def supports(self, file_path: Union[str, Path]) -> bool:
        """检查是否为Markdown文件"""
        return Path(file_path).suffix.lower() in self.supported_extensions
    
    def load(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        加载Markdown文档
        
        Args:
            file_path: Markdown文件路径
            
        Returns:
            文档内容和元数据
        """
        try:
            file_path = Path(file_path)
            
            # 检查文件是否存在
            if not file_path.exists():
                raise FileNotFoundError(f"文件不存在: {file_path}")
            
            # 获取文件信息
            file_stat = file_path.stat()
            file_hash = calculate_file_hash(file_path)
            
            # 读取文件内容
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
            
            # 提取元数据
            metadata = extract_metadata(content)
            
            return {
                'content': content,
                'metadata': {
                    **metadata,
                    'filename': file_path.name,
                    'file_path': str(file_path.absolute()),
                    'file_size': file_stat.st_size,
                    'file_hash': file_hash,
                    'content_type': 'text/markdown',
                    'loader': 'MarkdownLoader'
                }
            }
            
        except UnicodeDecodeError:
            # 尝试其他编码
            try:
                with open(file_path, 'r', encoding='gbk') as file:
                    content = file.read()
                logger.info(f"使用GBK编码读取文件: {file_path}")
            except UnicodeDecodeError:
                with open(file_path, 'r', encoding='latin-1') as file:
                    content = file.read()
                logger.info(f"使用latin-1编码读取文件: {file_path}")
            
            metadata = extract_metadata(content)
            return {
                'content': content,
                'metadata': {
                    **metadata,
                    'filename': file_path.name,
                    'file_path': str(file_path.absolute()),
                    'file_size': file_stat.st_size,
                    'file_hash': file_hash,
                    'content_type': 'text/markdown',
                    'loader': 'MarkdownLoader'
                }
            }
        except Exception as e:
            logger.error(f"加载Markdown文件失败 {file_path}: {e}")
            raise


class TextLoader(BaseLoader):
    """纯文本文档加载器"""
    
    def __init__(self):
        self.supported_extensions = {'.txt', '.text'}
    
    def supports(self, file_path: Union[str, Path]) -> bool:
        """检查是否为文本文件"""
        return Path(file_path).suffix.lower() in self.supported_extensions
    
    def load(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        加载纯文本文档
        
        Args:
            file_path: 文本文件路径
            
        Returns:
            文档内容和元数据
        """
        try:
            file_path = Path(file_path)
            
            # 检查文件是否存在
            if not file_path.exists():
                raise FileNotFoundError(f"文件不存在: {file_path}")
            
            # 获取文件信息
            file_stat = file_path.stat()
            file_hash = calculate_file_hash(file_path)
            
            # 读取文件内容
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
            
            # 提取元数据
            metadata = extract_metadata(content)
            
            return {
                'content': content,
                'metadata': {
                    **metadata,
                    'filename': file_path.name,
                    'file_path': str(file_path.absolute()),
                    'file_size': file_stat.st_size,
                    'file_hash': file_hash,
                    'content_type': 'text/plain',
                    'loader': 'TextLoader'
                }
            }
            
        except UnicodeDecodeError:
            # 尝试其他编码
            try:
                with open(file_path, 'r', encoding='gbk') as file:
                    content = file.read()
                logger.info(f"使用GBK编码读取文件: {file_path}")
            except UnicodeDecodeError:
                with open(file_path, 'r', encoding='latin-1') as file:
                    content = file.read()
                logger.info(f"使用latin-1编码读取文件: {file_path}")
            
            metadata = extract_metadata(content)
            return {
                'content': content,
                'metadata': {
                    **metadata,
                    'filename': file_path.name,
                    'file_path': str(file_path.absolute()),
                    'file_size': file_stat.st_size,
                    'file_hash': file_hash,
                    'content_type': 'text/plain',
                    'loader': 'TextLoader'
                }
            }
        except Exception as e:
            logger.error(f"加载文本文件失败 {file_path}: {e}")
            raise


class DocumentLoader:
    """文档加载器管理器"""
    
    def __init__(self):
        self.loaders = [
            PDFLoader(),
            MarkdownLoader(),
            TextLoader(),
        ]
        logger.info(f"初始化文档加载器，支持格式: {self.get_supported_extensions()}")
    
    def get_supported_extensions(self) -> List[str]:
        """获取支持的文件扩展名列表"""
        extensions = set()
        for loader in self.loaders:
            extensions.update(loader.supported_extensions)
        return sorted(list(extensions))
    
    def get_loader(self, file_path: Union[str, Path]) -> Optional[BaseLoader]:
        """
        根据文件路径获取合适的加载器
        
        Args:
            file_path: 文件路径
            
        Returns:
            合适的加载器，如果不支持则返回None
        """
        for loader in self.loaders:
            if loader.supports(file_path):
                return loader
        return None
    
    def load(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        加载文档
        
        Args:
            file_path: 文档路径
            
        Returns:
            文档内容和元数据
            
        Raises:
            ValueError: 不支持的文件格式
            FileNotFoundError: 文件不存在
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"文件不存在: {file_path}")
        
        loader = self.get_loader(file_path)
        if loader is None:
            supported = ', '.join(self.get_supported_extensions())
            raise ValueError(f"不支持的文件格式: {file_path.suffix}，支持的格式: {supported}")
        
        logger.info(f"使用 {loader.__class__.__name__} 加载文件: {file_path}")
        return loader.load(file_path)
    
    def load_multiple(self, file_paths: List[Union[str, Path]]) -> List[Dict[str, Any]]:
        """
        批量加载多个文档
        
        Args:
            file_paths: 文件路径列表
            
        Returns:
            文档列表
        """
        documents = []
        for file_path in file_paths:
            try:
                doc = self.load(file_path)
                documents.append(doc)
            except Exception as e:
                logger.error(f"加载文件失败 {file_path}: {e}")
                continue
        
        logger.info(f"批量加载完成: 成功 {len(documents)}/{len(file_paths)} 个文件")
        return documents 
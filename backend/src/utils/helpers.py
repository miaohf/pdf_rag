"""
辅助函数模块

提供各种通用的辅助函数。
"""

import hashlib
import json
import re
from typing import Any, Dict, List, Optional, Union
from pathlib import Path
from datetime import datetime


def calculate_file_hash(file_path: Union[str, Path]) -> str:
    """
    计算文件的MD5哈希值
    
    Args:
        file_path: 文件路径
        
    Returns:
        文件的MD5哈希值
    """
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def clean_text_for_technical_docs(text: str) -> str:
    """
    专门为技术文档设计的文本清理函数
    保留重要的结构信息，但去除影响向量匹配的噪音
    
    Args:
        text: 原始文本
        
    Returns:
        清理后的文本
    """
    if not text:
        return ""
    
    import re
    
    # 1. 移除HTML标签但保留内容
    text = re.sub(r'<[^>]+>', ' ', text)
    
    # 2. 移除多余的空白字符，但保留单个换行符（保持段落结构）
    text = re.sub(r'[ \t]+', ' ', text)  # 多个空格/制表符 -> 单个空格
    text = re.sub(r'\n\s*\n', '\n', text)  # 多个换行符 -> 单个换行符
    
    # 3. 保留重要的标点符号和数字格式
    # 不删除：. , ; : - / ( ) [ ] 等对技术文档重要的符号
    
    # 4. 移除一些不必要的特殊字符，但保留技术文档常见的符号
    text = re.sub(r'[^\w\s\u4e00-\u9fff。，！？；：""''（）【】《》\.\,\;\:\-\/\(\)\[\]\_]', ' ', text)
    
    # 5. 清理多余的空格
    text = re.sub(r' +', ' ', text)
    
    # 6. 清理开头和结尾的空白
    text = text.strip()
    
    return text


def clean_text(text: str) -> str:
    """
    清理文本内容
    
    Args:
        text: 原始文本
        
    Returns:
        清理后的文本
    """
    if not text:
        return ""
    
    # 移除多余的空白字符
    text = re.sub(r'\s+', ' ', text)
    
    # 移除特殊字符
    text = re.sub(r'[^\w\s\u4e00-\u9fff。，！？；：""''（）【】《》]', ' ', text)
    
    # 移除多余的空格
    text = ' '.join(text.split())
    
    return text.strip()


def format_file_size(size_bytes: int) -> str:
    """
    格式化文件大小
    
    Args:
        size_bytes: 字节数
        
    Returns:
        格式化的文件大小字符串
    """
    if size_bytes == 0:
        return "0B"
    
    size_names = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes /= 1024.0
        i += 1
    
    return f"{size_bytes:.1f}{size_names[i]}"


def sanitize_filename(filename: str) -> str:
    """
    清理文件名，移除不安全字符
    
    Args:
        filename: 原始文件名
        
    Returns:
        清理后的文件名
    """
    # 移除危险字符
    filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
    
    # 移除控制字符
    filename = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', filename)
    
    # 限制长度
    if len(filename) > 255:
        name, ext = Path(filename).stem, Path(filename).suffix
        filename = name[:255-len(ext)] + ext
    
    return filename


def extract_metadata(text: str) -> Dict[str, Any]:
    """
    从文本中提取元数据
    
    Args:
        text: 文本内容
        
    Returns:
        提取的元数据
    """
    metadata = {}
    
    # 文本长度
    metadata['length'] = len(text)
    metadata['word_count'] = len(text.split())
    
    # 检测语言（简单判断）
    chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
    english_chars = len(re.findall(r'[a-zA-Z]', text))
    
    if chinese_chars > english_chars:
        metadata['language'] = 'zh'
    else:
        metadata['language'] = 'en'
    
    # 提取可能的标题
    lines = text.split('\n')
    potential_titles = []
    for line in lines[:10]:  # 只检查前10行
        line = line.strip()
        if line and len(line) < 100:
            potential_titles.append(line)
    
    if potential_titles:
        metadata['potential_title'] = potential_titles[0]
    
    return metadata


def validate_config(config_dict: Dict[str, Any]) -> List[str]:
    """
    验证配置文件
    
    Args:
        config_dict: 配置字典
        
    Returns:
        错误信息列表
    """
    errors = []
    
    # 检查必需的配置项
    required_sections = ['database', 'ollama', 'document']
    for section in required_sections:
        if section not in config_dict:
            errors.append(f"缺少必需的配置节: {section}")
    
    # 检查数据库配置
    if 'database' in config_dict:
        db_config = config_dict['database']
        required_db_keys = ['host', 'port', 'name', 'user', 'password']
        for key in required_db_keys:
            if key not in db_config:
                errors.append(f"数据库配置缺少: {key}")
    
    # 检查Ollama配置
    if 'ollama' in config_dict:
        ollama_config = config_dict['ollama']
        required_ollama_keys = ['host', 'port', 'model']
        for key in required_ollama_keys:
            if key not in ollama_config:
                errors.append(f"Ollama配置缺少: {key}")
    
    return errors


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    """
    将文本分块
    
    Args:
        text: 要分块的文本
        chunk_size: 块大小
        overlap: 重叠大小
        
    Returns:
        文本块列表
    """
    if not text or chunk_size <= 0:
        return []
    
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        
        # 如果不是最后一块，尝试在句号处分割
        if end < len(text):
            # 查找最近的句号
            for i in range(end, max(start + chunk_size // 2, end - 100), -1):
                if text[i] in '。！？.!?':
                    end = i + 1
                    break
        
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        
        # 下一块的开始位置
        start = max(start + chunk_size - overlap, end)
        
        # 防止无限循环
        if start <= chunks[-1:] and len(chunks) > 1:
            break
    
    return chunks


def format_datetime(dt: Optional[datetime] = None) -> str:
    """
    格式化日期时间
    
    Args:
        dt: 日期时间对象，默认为当前时间
        
    Returns:
        格式化的日期时间字符串
    """
    if dt is None:
        dt = datetime.now()
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def safe_json_loads(json_str: str, default: Any = None) -> Any:
    """
    安全地解析JSON字符串
    
    Args:
        json_str: JSON字符串
        default: 解析失败时的默认值
        
    Returns:
        解析结果或默认值
    """
    try:
        return json.loads(json_str)
    except (json.JSONDecodeError, TypeError):
        return default


def safe_json_dumps(obj: Any, default: str = "{}") -> str:
    """
    安全地序列化为JSON字符串
    
    Args:
        obj: 要序列化的对象
        default: 序列化失败时的默认值
        
    Returns:
        JSON字符串或默认值
    """
    try:
        return json.dumps(obj, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return default 
"""
日志工具模块

提供统一的日志配置和管理功能。
"""

import logging
import sys
from pathlib import Path
from typing import Optional


class Logger:
    """日志管理器"""
    
    _loggers = {}
    _initialized = False
    
    @classmethod
    def setup(cls, config: Optional[object] = None) -> None:
        """
        设置全局日志配置
        
        Args:
            config: 配置对象
        """
        if cls._initialized:
            return
            
        if config is None:
            # 延迟导入避免循环依赖
            from src.utils.config import Config
            config = Config()
        
        # 创建日志目录
        log_file = Path(config.logging.file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        # 配置根日志器
        root_logger = logging.getLogger()
        root_logger.setLevel(getattr(logging, config.logging.level.upper()))
        
        # 清除现有处理器
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
        
        # 创建格式器
        formatter = logging.Formatter(config.logging.format)
        
        # 创建控制台处理器
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)
        
        # 创建文件处理器
        file_handler = logging.FileHandler(
            config.logging.file, 
            encoding='utf-8',
            mode='a'
        )
        file_handler.setLevel(getattr(logging, config.logging.level.upper()))
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
        
        cls._initialized = True
    
    @classmethod
    def get_logger(cls, name: str) -> logging.Logger:
        """
        获取指定名称的日志器
        
        Args:
            name: 日志器名称
            
        Returns:
            日志器实例
        """
        if not cls._initialized:
            cls.setup()
        
        if name not in cls._loggers:
            logger = logging.getLogger(name)
            cls._loggers[name] = logger
        
        return cls._loggers[name]


# 便捷函数
def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    获取日志器的便捷函数
    
    Args:
        name: 日志器名称，默认使用调用模块名
        
    Returns:
        日志器实例
    """
    if name is None:
        # 获取调用者的模块名
        frame = sys._getframe(1)
        name = frame.f_globals.get('__name__', 'unknown')
    
    return Logger.get_logger(name)


# 创建默认日志器
logger = get_logger('pdf_rag') 
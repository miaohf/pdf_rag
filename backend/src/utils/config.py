"""
配置管理模块

负责加载和管理系统配置，支持YAML配置文件和环境变量。
"""

import os
import yaml
from typing import Dict, Any, Optional
from pathlib import Path
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


class DatabaseConfig(BaseModel):
    """数据库配置"""
    host: str = "127.0.0.1"
    port: int = 5432
    name: str = "pdf_rag"
    user: str = "postgres"
    password: str = "postgres"
    pool_size: int = 10
    max_overflow: int = 20
    
    @property
    def url(self) -> str:
        """获取数据库连接URL"""
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.name}"


class OllamaConfig(BaseModel):
    """Ollama配置"""
    host: str = "192.168.31.80"
    port: int = 11434
    model: str = "qwen3:14b"
    timeout: int = 600
    temperature: float = 0.7
    max_tokens: int = 8192
    
    @property
    def url(self) -> str:
        """获取Ollama API URL"""
        return f"http://{self.host}:{self.port}"


class EmbeddingConfig(BaseModel):
    """嵌入模型配置"""
    model: str = "nomic-embed-text"
    batch_size: int = 64
    max_length: int = 1024
    api_url: str = "http://192.168.31.80:11434"
    use_remote_api: bool = True


class DocumentConfig(BaseModel):
    """文档处理配置"""
    chunk_size: int = 600
    chunk_overlap: int = 100
    max_chunks_per_document: int = 200


class RetrievalConfig(BaseModel):
    """向量检索配置"""
    top_k: int = 12
    similarity_threshold: float = 0.5
    rerank: bool = True
    rerank_top_k: int = 8


class LoggingConfig(BaseModel):
    """日志配置"""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file: str = "logs/pdf_rag.log"


class Settings(BaseSettings):
    """环境变量配置"""
    # 数据库
    db_host: str = Field(default="127.0.0.1", env="DB_HOST")
    db_port: int = Field(default=5432, env="DB_PORT")
    db_name: str = Field(default="pdf_rag", env="DB_NAME")
    db_user: str = Field(default="postgres", env="DB_USER")
    db_pass: str = Field(default="postgres", env="DB_PASS")
    
    # Ollama
    ollama_host: str = Field(default="192.168.31.80", env="OLLAMA_HOST")
    ollama_port: int = Field(default=11434, env="OLLAMA_PORT")
    ollama_model: str = Field(default="qwen3:14b", env="OLLAMA_MODEL")
    
    # 应用
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    log_file: str = Field(default="logs/pdf_rag.log", env="LOG_FILE")
    
    # API
    api_host: str = Field(default="0.0.0.0", env="API_HOST")
    api_port: int = Field(default=8000, env="API_PORT")
    
    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore"  # 允许额外的字段
    }


class Config:
    """主配置类"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        初始化配置
        
        Args:
            config_path: 配置文件路径，默认为 config/config.yaml
        """
        self.config_path = config_path or "config/config.yaml"
        self._config_data = {}
        self._settings = Settings()
        
        # 加载配置
        self._load_config()
        self._merge_env_vars()
        
        # 初始化各模块配置
        self.database = self._init_database_config()
        self.ollama = self._init_ollama_config()
        self.embedding = self._init_embedding_config()
        self.document = self._init_document_config()
        self.retrieval = self._init_retrieval_config()
        self.logging = self._init_logging_config()
    
    def _load_config(self) -> None:
        """加载YAML配置文件"""
        config_file = Path(self.config_path)
        if config_file.exists():
            with open(config_file, 'r', encoding='utf-8') as f:
                self._config_data = yaml.safe_load(f) or {}
        else:
            print(f"配置文件 {self.config_path} 不存在，使用默认配置")
    
    def _merge_env_vars(self) -> None:
        """合并环境变量配置"""
        # 数据库配置
        if 'database' not in self._config_data:
            self._config_data['database'] = {}
        
        db_config = self._config_data['database']
        db_config['host'] = self._settings.db_host
        db_config['port'] = self._settings.db_port
        db_config['name'] = self._settings.db_name
        db_config['user'] = self._settings.db_user
        db_config['password'] = self._settings.db_pass
        
        # Ollama配置
        if 'ollama' not in self._config_data:
            self._config_data['ollama'] = {}
            
        ollama_config = self._config_data['ollama']
        ollama_config['host'] = self._settings.ollama_host
        ollama_config['port'] = self._settings.ollama_port
        ollama_config['model'] = self._settings.ollama_model
        
        # 日志配置
        if 'logging' not in self._config_data:
            self._config_data['logging'] = {}
            
        log_config = self._config_data['logging']
        log_config['level'] = self._settings.log_level
        log_config['file'] = self._settings.log_file
    
    def _init_database_config(self) -> DatabaseConfig:
        """初始化数据库配置"""
        db_data = self._config_data.get('database', {})
        return DatabaseConfig(**db_data)
    
    def _init_ollama_config(self) -> OllamaConfig:
        """初始化Ollama配置"""
        ollama_data = self._config_data.get('ollama', {})
        return OllamaConfig(**ollama_data)
    
    def _init_embedding_config(self) -> EmbeddingConfig:
        """初始化嵌入模型配置"""
        embedding_data = self._config_data.get('embedding', {})
        return EmbeddingConfig(**embedding_data)
    
    def _init_document_config(self) -> DocumentConfig:
        """初始化文档处理配置"""
        doc_data = self._config_data.get('document', {})
        return DocumentConfig(**doc_data)
    
    def _init_retrieval_config(self) -> RetrievalConfig:
        """初始化检索配置"""
        retrieval_data = self._config_data.get('retrieval', {})
        return RetrievalConfig(**retrieval_data)
    
    def _init_logging_config(self) -> LoggingConfig:
        """初始化日志配置"""
        log_data = self._config_data.get('logging', {})
        return LoggingConfig(**log_data)
    
    def get(self, key: str, default: Any = None) -> Any:
        """获取配置值"""
        keys = key.split('.')
        value = self._config_data
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'database': self.database.model_dump(),
            'ollama': self.ollama.model_dump(),
            'embedding': self.embedding.model_dump(),
            'document': self.document.model_dump(),
            'retrieval': self.retrieval.model_dump(),
            'logging': self.logging.model_dump(),
        } 
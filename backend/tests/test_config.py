"""
配置模块测试
"""

import pytest
from src.utils.config import Config, DatabaseConfig, OllamaConfig


class TestConfig:
    """配置测试类"""
    
    def test_config_initialization(self):
        """测试配置初始化"""
        config = Config()
        
        assert config is not None
        assert config.database is not None
        assert config.ollama is not None
        assert config.embedding is not None
        assert config.document is not None
        assert config.retrieval is not None
        assert config.logging is not None
    
    def test_database_config(self):
        """测试数据库配置"""
        config = Config()
        db_config = config.database
        
        assert isinstance(db_config, DatabaseConfig)
        assert db_config.host is not None
        assert db_config.port > 0
        assert db_config.name is not None
        assert db_config.user is not None
        assert db_config.password is not None
        
        # 测试URL生成
        url = db_config.url
        assert "postgresql://" in url
        assert str(db_config.port) in url
    
    def test_ollama_config(self):
        """测试Ollama配置"""
        config = Config()
        ollama_config = config.ollama
        
        assert isinstance(ollama_config, OllamaConfig)
        assert ollama_config.host is not None
        assert ollama_config.port > 0
        assert ollama_config.model is not None
        
        # 测试URL生成
        url = ollama_config.url
        assert "http://" in url
        assert str(ollama_config.port) in url
    
    def test_config_to_dict(self):
        """测试配置转字典"""
        config = Config()
        config_dict = config.to_dict()
        
        assert isinstance(config_dict, dict)
        assert 'database' in config_dict
        assert 'ollama' in config_dict
        assert 'embedding' in config_dict
        assert 'document' in config_dict
        assert 'retrieval' in config_dict
        assert 'logging' in config_dict
    
    def test_config_get_method(self):
        """测试配置获取方法"""
        config = Config()
        
        # 测试存在的配置
        db_host = config.get('database.host')
        assert db_host is not None
        
        # 测试不存在的配置
        non_existent = config.get('non.existent.key', 'default')
        assert non_existent == 'default' 
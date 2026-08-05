# PDF RAG Backend

基于检索增强生成（RAG）技术的法律法规智能问答系统后端，提供高性能的文档处理和智能问答服务。

## ✨ 核心特性

- 🤖 **智能问答**: 基于本地大语言模型的上下文感知回答
- 🔍 **语义检索**: PostgreSQL + pgvector 的高效向量检索
- 🧠 **统一RAG引擎**: 查询扩展、层次检索、重排序与答案生成的一体化管线
- 📚 **文档处理**: 支持 PDF 等多格式文档的智能分片
- 🔄 **Agentic 策略**: 可选多步推理与工具调用扩展
- 🔒 **本地部署**: 完全本地化，保护数据隐私
- ⚡ **高性能**: RTX 4090 显卡优化，支持大规模文档库
- 🌐 **RESTful API**: 标准化的 Web API 接口

## 🏗️ 技术架构

### 系统架构图
```
PDF文档 → 文档处理 → 向量化 → PostgreSQL+pgvector
                                    ↓
用户查询 → 语义检索 → RAG增强 → Qwen3:14b → 智能回答
```

### 核心技术栈
- **框架**: FastAPI + Python 3.12+
- **数据库**: PostgreSQL + pgvector 扩展
- **大语言模型**: Ollama (Qwen3:14b)
- **嵌入模型**: Qwen3 Embedding
- **文档处理**: LangChain
- **包管理**: uv (国内镜像优化)

## 🚀 快速开始

### 环境要求
- Python 3.12+
- PostgreSQL 13+ (with pgvector)
- Ollama (已安装 Qwen3:14b 模型)
- 推荐: NVIDIA RTX 4090 24GB 显卡

### 快速启动
```bash
# 克隆项目
git clone <repository-url>
cd pdf_rag/backend

# 快速启动脚本
chmod +x quick_start.sh
./quick_start.sh
```

### 手动安装
```bash
# 安装 uv 包管理器
curl -LsSf https://astral.sh/uv/install.sh | sh

# 同步依赖
uv sync

# 激活虚拟环境
source .venv/bin/activate

# 配置环境变量
cp env_example .env
# 编辑 .env 文件配置数据库和 Ollama
```

### 数据库设置
```sql
-- 创建数据库
CREATE DATABASE pdf_rag;

-- 切换到数据库
\c pdf_rag

-- 安装 pgvector 扩展
CREATE EXTENSION IF NOT EXISTS vector;
```

### 初始化数据库
```bash
python main.py init-db
```

## 📁 项目结构

```
backend/
├── app.py                    # Web API 服务入口
├── main.py                   # 命令行工具
├── quick_start.sh           # 快速启动脚本
├── pyproject.toml           # uv 项目配置
├── .env.example             # 环境变量模板
├── config/
│   ├── config.yaml         # 主配置文件
│   └── rtx4090_optimization.md # 显卡优化说明
├── src/
│   ├── core/               # 核心模块
│   │   ├── database.py    # 数据库连接
│   │   ├── embeddings.py  # 嵌入模型
│   │   ├── llm.py         # 大语言模型
│   │   └── vector_store.py # 向量存储
│   ├── document/           # 文档处理
│   │   ├── processor.py   # 文档处理器
│   │   ├── chunker.py     # 文档分片
│   │   └── loader.py      # 文档加载
│   ├── rag/               # RAG 引擎
│   │   ├── engine.py      # 统一检索-生成引擎
│   │   ├── retriever.py   # 层次化检索器
│   │   ├── service.py     # RAG 服务层
│   │   └── agentic_engine.py # Agentic 推理扩展
│   ├── api/               # Web API
│   │   ├── routes.py      # 路由定义
│   │   ├── models.py      # 数据模型
│   │   └── middleware.py  # 中间件
│   └── utils/             # 工具模块
├── docs/                   # 文档存储目录
├── migrations/             # 数据库迁移
├── tests/                  # 测试用例
└── logs/                   # 日志目录
```

## 🔧 使用方法

### Web API 服务
```bash
# 启动 API 服务
python app.py
# 或
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

访问 [http://localhost:8000](http://localhost:8000) 查看 API 文档

### 命令行工具
```bash
# 检查服务状态
python main.py --check

# 导入文档
python main.py --import docs/

# 单次查询
python main.py --query "什么是远程控制停车系统？"

# 交互式查询
python main.py --interactive

# 查看统计信息
python main.py --stats
```

## 🌐 API 接口

### 核心接口
- `GET /health` - 健康检查
- `POST /api/query` - 智能问答
- `POST /api/query/stream` - 流式问答
- `POST /api/documents/import` - 导入文档
- `GET /api/documents/search` - 搜索文档
- `GET /api/documents/preview/{filename}` - 文档预览
- `GET /api/documents/download/{filename}` - 文档下载
- `GET /api/stats` - 系统统计

### 问答接口示例
```bash
curl -X POST "http://localhost:8000/api/query" \
     -H "Content-Type: application/json" \
     -d '{"question": "什么是远程控制停车系统？"}'
```

## ⚙️ 配置说明

### 环境变量
```env
# 数据库配置
DB_HOST=localhost
DB_PORT=5432
DB_NAME=pdf_rag
DB_USER=postgres
DB_PASSWORD=your_password

# Ollama 配置
OLLAMA_HOST=127.0.0.1
OLLAMA_PORT=11434
OLLAMA_MODEL=qwen3:14b
```

### 主配置文件
```yaml
database:
  db_host: localhost
  db_port: 5432
  db_name: pdf_rag

ollama:
  host: 127.0.0.1
  port: 11434
  model: qwen3:14b

rag:
  chunk_size: 2000
  chunk_overlap: 200
  top_k: 8
```

## 🎯 性能优化

### RTX 4090 24GB 显卡优化
- **最大 Token 数**: 8192 (支持更长上下文)
- **批处理大小**: 64 (提高并行处理能力)
- **文档分片**: 2000 字符 (保持完整语义)
- **检索数量**: Top-8 (提高回答准确性)

### 向量数据库优化
- **索引策略**: IVFFlat 索引加速检索
- **相似度计算**: 余弦相似度匹配
- **缓存机制**: 热点文档缓存

## 🔍 RAG 引擎

### 文档处理流程
1. **文档加载**: 支持 PDF、TXT、MD 等格式
2. **智能分片**: 基于语义的文档分片
3. **向量化**: 使用 Qwen3 Embedding 生成向量
4. **存储**: PostgreSQL + pgvector 高效存储

### 检索增强生成
1. **查询理解**: 用户问题的语义分析
2. **向量检索**: 基于相似度的文档检索
3. **上下文构建**: 相关文档片段组合
4. **答案生成**: LLM 基于上下文生成回答

## 🛠️ 开发特性

### 错误处理
- 完善的异常捕获和处理
- 友好的错误信息返回
- 日志记录和监控

### 扩展性
- 模块化设计，易于扩展
- 支持多种文档格式
- 可配置的模型参数

### 监控和日志
- 详细的操作日志
- 性能监控指标
- 错误追踪和分析

## 📊 系统监控

### 健康检查
- 数据库连接状态
- Ollama 服务状态
- 系统资源使用

### 性能指标
- 查询响应时间
- 文档检索准确率
- 系统并发处理能力

---

**设计理念**: 高性能、高可用、易扩展的企业级 AI 问答系统，为法律法规查询提供准确可靠的智能服务。




# PDF RAG - 法律法规智能问答系统

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-green.svg)](https://fastapi.tiangolo.com/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-13+-blue.svg)](https://www.postgresql.org/)
[![Next.js 15](https://img.shields.io/badge/Next.js-15-black.svg)](https://nextjs.org/)

基于检索增强生成（RAG）技术的法律法规智能问答系统，提供高性能的文档处理和智能问答服务。

## ✨ 核心特性

### 🤖 智能问答能力
- **本地大语言模型**：基于 Ollama + Qwen3:14b 的上下文感知回答
- **Agentic RAG**：智能代理式推理，支持复杂查询分解和多步推理
- **多轮对话**：完整的对话状态管理和上下文维护
- **工具调用**：集成计算器、文档搜索等专业工具

### 🔍 高级检索技术
- **分层检索**：父子分片两阶段检索，平衡精度与上下文
- **向量检索**：PostgreSQL + pgvector 的高效向量检索
- **多维重排**：语义相似度 + 关键词匹配 + 内容质量综合评分
- **智能过滤**：基于语义分析的内容相关性过滤

### 📚 文档处理
- **多格式支持**：PDF、Word、Markdown 等多格式文档处理
- **智能分片**：层次化文档分片，保持语义完整性
- **向量化存储**：高效的文档嵌入和向量存储

### 🔒 企业级特性
- **本地部署**：完全本地化，保护数据隐私
- **高性能**：RTX 4090 显卡优化，支持大规模文档库
- **RESTful API**：标准化的 Web API 接口
- **现代化 UI**：响应式前端界面，支持暗色主题

## 🏗️ 系统架构

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   前端界面      │    │   后端服务      │    │   数据存储      │
│  Next.js 15     │◄──►│   FastAPI       │◄──►│  PostgreSQL     │
│  TypeScript     │    │   Python 3.12   │    │  + pgvector     │
│  Tailwind CSS   │    │   Ollama        │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         │                       │                       │
    ┌────▼────┐              ┌───▼───┐              ┌────▼────┐
    │ 用户交互 │              │ RAG引擎│              │ 向量库  │
    │ 文档查看 │              │ 智能推理│              │ 文档库  │
    │ 搜索高亮 │              │ 工具调用│              │ 会话库  │
    └─────────┘              └───────┘              └─────────┘
```

### 核心组件架构
```
PDF文档 → 文档处理 → 智能分片 → 向量化 → PostgreSQL+pgvector
                                              ↓
用户查询 → 查询处理 → 分层检索 → 多维重排 → Agentic推理 → 智能回答
```

## 🚀 快速开始

### 环境要求

**基础环境**
- Python 3.12+
- Node.js 18+
- PostgreSQL 13+ (with pgvector)
- Ollama

**推荐硬件**
- NVIDIA RTX 4090 24GB 显卡
- 32GB+ 内存
- 100GB+ 存储空间

### 一键启动

```bash
# 克隆项目
git clone <repository-url>
cd pdf_rag

# 后端快速启动
cd backend
chmod +x quick_start.sh
./quick_start.sh

# 前端启动
cd ../frontend
npm install
npm run dev
```

### 详细安装步骤

#### 1. 后端环境设置

```bash
cd backend

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

#### 2. 数据库设置

```sql
-- 创建数据库
CREATE DATABASE pdf_rag;

-- 切换到数据库
\c pdf_rag

-- 安装 pgvector 扩展
CREATE EXTENSION IF NOT EXISTS vector;
```

#### 3. Ollama 设置

```bash
# 安装 Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# 下载 Qwen 模型
ollama pull qwen2:14b

# 启动 Ollama 服务
ollama serve
```

#### 4. 初始化数据库

```bash
# 运行数据库迁移
python main.py init-db

# 启动后端服务
python app.py
```

#### 5. 前端设置

```bash
cd frontend

# 安装依赖
npm install

# 创建环境配置
cp .env.example .env.local
# 编辑 .env.local 配置后端 API 地址

# 启动开发服务器
npm run dev
```

## 📁 项目结构

```
pdf_rag/
├── backend/                    # 后端服务
│   ├── src/                   # 源代码
│   │   ├── core/             # 核心组件
│   │   │   ├── database.py   # 数据库连接
│   │   │   ├── embeddings.py # 嵌入模型
│   │   │   ├── llm.py        # 大语言模型
│   │   │   ├── models.py     # 数据模型
│   │   │   ├── vector_store.py # 向量存储
│   │   │   └── hierarchical_retriever.py # 分层检索器
│   │   ├── document/         # 文档处理
│   │   │   ├── loader.py     # 文档加载器
│   │   │   ├── chunker.py    # 文档分片器
│   │   │   └── processor.py  # 文档处理器
│   │   ├── rag/              # RAG 核心
│   │   │   ├── service.py    # RAG 服务层
│   │   │   ├── agentic_engine.py # Agentic 引擎
│   │   │   ├── conversation.py # 对话管理
│   │   │   ├── content_filter.py # 内容过滤
│   │   │   └── tools.py      # 工具调用
│   │   └── utils/            # 工具模块
│   ├── app.py                # Web API 入口
│   ├── main.py               # 命令行工具
│   ├── pyproject.toml        # 项目配置
│   └── quick_start.sh        # 快速启动脚本
├── frontend/                  # 前端应用
│   ├── src/                  # 源代码
│   │   ├── app/             # Next.js App Router
│   │   ├── components/      # React 组件
│   │   │   ├── ui/         # 基础 UI 组件
│   │   │   ├── chat/       # 聊天界面
│   │   │   └── document/   # 文档查看器
│   │   └── lib/            # 工具库
│   ├── package.json         # 依赖配置
│   └── next.config.js       # Next.js 配置
└── README.md                 # 项目说明
```

## 🔧 核心技术栈

### 后端技术栈
- **框架**: FastAPI + Python 3.12+
- **数据库**: PostgreSQL + pgvector 扩展
- **大语言模型**: Ollama (Qwen3:14b)
- **嵌入模型**: Qwen3 Embedding
- **文档处理**: LangChain
- **包管理**: uv (国内镜像优化)

### 前端技术栈
- **框架**: Next.js 15 + TypeScript
- **样式**: Tailwind CSS + CSS Variables
- **组件**: 自定义 UI 组件库
- **图标**: Lucide React
- **状态管理**: React Hooks + Context API

## 🎯 核心功能详解

### 智能问答系统
- **实时对话**: 与后端 AI 的实时问答交互
- **思考过程**: AI 分析过程的可视化展示
- **源文档引用**: 显示答案来源和相似度评分
- **多轮对话**: 完整的对话历史记录和上下文维护

### 分层检索技术
- **子分片精准检索**: 在细粒度分片中精确匹配
- **父分片上下文扩展**: 提供更丰富的语义上下文
- **智能合并去重**: 检测并去除重叠内容
- **多策略支持**: child_to_parent、parent_to_child、hybrid

### 文档处理流程
1. **文档解析**: 支持 PDF、Word、Markdown 等格式
2. **智能分片**: 基于语义的层次化分片
3. **向量化**: 使用 Qwen3 嵌入模型生成向量
4. **存储索引**: PostgreSQL + pgvector 高效存储

### Agentic RAG 引擎
- **查询规划**: 自动分解复杂查询为子问题
- **多步推理**: 逐步推理并记录思考过程
- **工具调用**: 集成计算器、搜索等专业工具
- **结果合成**: 智能合成最终答案

## 📊 性能特点

- **检索速度**: pgvector 向量检索 < 100ms
- **推理性能**: RTX 4090 优化，支持大规模并发
- **存储效率**: 1GB 文档约占用 2GB 存储空间
- **响应时间**: 平均问答响应时间 < 3s

## 🔨 开发指南

### API 接口

#### 问答接口
```http
POST /api/query
Content-Type: application/json

{
  "question": "什么是Remote Control Parking的最大速度限制？",
  "session_id": "optional_session_id",
  "top_k": 5,
  "similarity_threshold": 0.3
}
```

#### 文档上传
```http
POST /api/documents/upload
Content-Type: multipart/form-data

file: <document_file>
```

#### 对话历史
```http
GET /api/conversations/{session_id}/history
```

### 配置选项

#### 检索配置
```yaml
retrieval:
  top_k: 12                    # 初始检索数量
  similarity_threshold: 0.5    # 相似度阈值
  rerank: true                 # 启用重排序
  rerank_top_k: 8             # 重排序保留数量
```

#### 文档配置
```yaml
document:
  chunk_size: 512             # 分片大小
  chunk_overlap: 128          # 分片重叠
  use_hierarchical_chunking: true  # 启用层次分片
```

## 🤝 贡献指南

1. Fork 项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 打开 Pull Request

## 📝 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情

## 🆘 支持与反馈

- **问题反馈**: [GitHub Issues](https://github.com/your-repo/pdf_rag/issues)
- **功能建议**: [GitHub Discussions](https://github.com/your-repo/pdf_rag/discussions)
- **技术交流**: 欢迎加入项目讨论群

## 🙏 致谢

感谢以下开源项目的支持：
- [FastAPI](https://fastapi.tiangolo.com/) - 现代化的 Python Web 框架
- [Next.js](https://nextjs.org/) - React 全栈框架
- [Ollama](https://ollama.ai/) - 本地 LLM 运行环境
- [pgvector](https://github.com/pgvector/pgvector) - PostgreSQL 向量扩展
- [LangChain](https://langchain.readthedocs.io/) - LLM 应用开发框架

---

**PDF RAG** - 让法律法规智能问答触手可及 🚀

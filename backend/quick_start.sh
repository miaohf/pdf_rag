#!/bin/bash

# PDF RAG 快速启动脚本
# 作者: PDF RAG Team
# 版本: v0.1.0

set -e

echo "🚀 PDF RAG 快速启动脚本"
echo "========================"

# 检查系统要求
check_requirements() {
    echo "📋 检查系统要求..."
    
    # 检查Python版本
    if ! command -v python3.12 &> /dev/null && ! command -v python3 &> /dev/null; then
        echo "❌ 错误: 需要Python 3.12+，请先安装Python"
        exit 1
    fi
    
    # 检查PostgreSQL
    if ! command -v psql &> /dev/null; then
        echo "⚠️  警告: 未检测到PostgreSQL，请确保数据库已安装并运行"
    fi
    
    echo "✅ 系统要求检查完成"
}

# 安装uv包管理器
install_uv() {
    if ! command -v uv &> /dev/null; then
        echo "📦 安装uv包管理器..."
        curl -LsSf https://astral.sh/uv/install.sh | sh
        source $HOME/.cargo/env
        echo "✅ uv安装完成"
    else
        echo "✅ uv已安装"
    fi
}

# 配置国内镜像源
setup_mirrors() {
    echo "🌐 配置国内镜像源..."
    export UV_INDEX_URL="https://pypi.tuna.tsinghua.edu.cn/simple"
    export UV_EXTRA_INDEX_URL="https://pypi.org/simple"
    echo "✅ 镜像源配置完成"
}

# 安装依赖
install_dependencies() {
    echo "📚 安装项目依赖..."
    uv sync
    echo "✅ 依赖安装完成"
}

# 创建必要目录
create_directories() {
    echo "📁 创建必要目录..."
    mkdir -p logs
    mkdir -p migrations
    echo "✅ 目录创建完成"
}

# 配置环境变量
setup_env() {
    if [ ! -f .env ]; then
        echo "⚙️  配置环境变量..."
        cp .env.example .env
        echo "✅ 环境变量文件已创建，请根据需要修改 .env 文件"
    else
        echo "✅ 环境变量文件已存在"
    fi
}

# 检查服务状态
check_services() {
    echo "🔍 检查服务状态..."
    
    # 激活虚拟环境并检查
    source .venv/bin/activate
    
    if python main.py --check; then
        echo "✅ 服务检查通过"
    else
        echo "⚠️  服务检查失败，请检查配置"
    fi
}

# 显示使用说明
show_usage() {
    echo ""
    echo "🎉 安装完成！"
    echo "=============="
    echo ""
    echo "📖 使用说明:"
    echo "1. 激活虚拟环境:"
    echo "   source .venv/bin/activate"
    echo ""
    echo "2. 导入文档:"
    echo "   python main.py --import docs/"
    echo ""
    echo "3. 开始问答:"
    echo "   python main.py --interactive"
    echo ""
    echo "4. 启动Web服务:"
    echo "   python app.py"
    echo ""
    echo "5. 查看帮助:"
    echo "   python main.py --help"
    echo ""
    echo "📚 更多信息请查看 README.md"
    echo ""
}

# 主执行流程
main() {
    check_requirements
    install_uv
    setup_mirrors
    install_dependencies
    create_directories
    setup_env
    check_services
    show_usage
}

# 错误处理
trap 'echo "❌ 安装过程中发生错误，请检查上述输出"; exit 1' ERR

# 运行主函数
main "$@" 
#!/usr/bin/env python3
"""
PDF RAG 命令行主程序

提供文档导入、查询、服务检查等功能的命令行接口。
"""

import click
import sys
import time
from pathlib import Path
from typing import Optional, List
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Prompt

from src.utils.config import Config
from src.utils.logger import Logger, get_logger
from src.core.database import Database

console = Console()
logger = get_logger(__name__)


def init_app() -> tuple[Config, Database]:
    """初始化应用"""
    try:
        # 加载配置
        config = Config()
        
        # 设置日志
        Logger.setup(config)
        
        # 初始化数据库
        db = Database(config)
        
        return config, db
        
    except Exception as e:
        console.print(f"[red]初始化失败: {e}[/red]")
        sys.exit(1)


@click.group(invoke_without_command=True)
@click.option('--check', is_flag=True, help='检查系统状态')
@click.option('--stats', is_flag=True, help='显示统计信息')
@click.option('--query', '-q', help='单次查询')
@click.option('--interactive', '-i', is_flag=True, help='交互式查询')
@click.option('--import', 'import_path', help='导入文档路径')
@click.option('--force', is_flag=True, help='强制重新处理已存在的文档')
@click.pass_context
def cli(ctx, check, stats, query, interactive, import_path, force):
    """PDF RAG - 法律法规智能问答系统"""
    
    # 显示欢迎信息
    if ctx.invoked_subcommand is None and not any([check, stats, query, interactive, import_path]):
        show_welcome()
        return
    
    # 初始化
    config, db = init_app()
    ctx.ensure_object(dict)
    ctx.obj['config'] = config
    ctx.obj['db'] = db
    
    # 执行相应操作
    if check:
        check_system(config, db)
    elif stats:
        show_stats(db)
    elif query:
        single_query(query, config)
    elif interactive:
        interactive_mode(config)
    elif import_path:
        import_documents(import_path, config, db, force)


def show_welcome():
    """显示欢迎信息"""
    welcome_text = """
[bold blue]PDF RAG - 法律法规智能问答系统[/bold blue]

🚀 基于Agentic RAG技术的智能文档问答系统
📚 专为法律法规文档设计，提供准确可靠的咨询服务

[bold yellow]使用方法:[/bold yellow]
  python main.py --help         # 查看帮助
  python main.py --check        # 检查系统状态
  python main.py --import docs/ # 导入文档
  python main.py --query "问题" # 单次查询
  python main.py --interactive  # 交互式问答
  python main.py --stats        # 查看统计信息

[bold yellow]Web服务:[/bold yellow]
  python app.py                 # 启动Web API服务

📖 更多信息请查看 README.md
    """
    console.print(Panel(welcome_text, title="欢迎使用", border_style="blue"))


def check_system(config: Config, db: Database):
    """检查系统状态"""
    console.print("\n[bold blue]🔍 系统状态检查[/bold blue]")
    
    checks = []
    
    # 检查数据库连接
    with console.status("[bold green]检查数据库连接..."):
        db_ok = db.check_connection()
        checks.append(("数据库连接", "✅ 正常" if db_ok else "❌ 失败"))
    
    # 检查Ollama连接
    with console.status("[bold green]检查Ollama服务..."):
        try:
            import httpx
            response = httpx.get(f"{config.ollama.url}/api/tags", timeout=5)
            ollama_ok = response.status_code == 200
            checks.append(("Ollama服务", "✅ 正常" if ollama_ok else "❌ 失败"))
        except Exception:
            checks.append(("Ollama服务", "❌ 连接失败"))
    
    # 检查模型可用性
    with console.status("[bold green]检查模型可用性..."):
        try:
            import httpx
            response = httpx.get(f"{config.ollama.url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get('models', [])
                model_names = [m['name'] for m in models]
                model_ok = config.ollama.model in model_names
                checks.append(("Qwen3模型", "✅ 可用" if model_ok else "❌ 未找到"))
            else:
                checks.append(("Qwen3模型", "❌ 无法检查"))
        except Exception:
            checks.append(("Qwen3模型", "❌ 检查失败"))
    
    # 检查数据库表
    with console.status("[bold green]检查数据库表..."):
        try:
            stats = db.get_stats()
            checks.append(("数据库表", "✅ 正常"))
        except Exception:
            checks.append(("数据库表", "❌ 错误"))
    
    # 显示检查结果
    table = Table(title="系统状态检查结果", show_header=True, header_style="bold magenta")
    table.add_column("检查项", style="cyan", no_wrap=True)
    table.add_column("状态", style="green")
    
    for check_name, status in checks:
        table.add_row(check_name, status)
    
    console.print(table)
    
    # 总体状态
    all_ok = all("✅" in status for _, status in checks)
    if all_ok:
        console.print("\n[bold green]✅ 系统运行正常[/bold green]")
    else:
        console.print("\n[bold red]❌ 系统存在问题，请检查配置[/bold red]")


def show_stats(db: Database):
    """显示统计信息"""
    console.print("\n[bold blue]📊 系统统计信息[/bold blue]")
    
    with console.status("[bold green]获取统计信息..."):
        stats = db.get_stats()
    
    if not stats:
        console.print("[red]无法获取统计信息[/red]")
        return
    
    # 创建统计表格
    table = Table(title="数据统计", show_header=True, header_style="bold magenta")
    table.add_column("项目", style="cyan", no_wrap=True)
    table.add_column("数量", style="green", justify="right")
    
    table.add_row("文档总数", str(stats.get('documents', 0)))
    table.add_row("文档片段", str(stats.get('chunks', 0)))
    table.add_row("查询历史", str(stats.get('queries', 0)))
    table.add_row("已处理文档", str(stats.get('processed_documents', 0)))
    table.add_row("处理率", stats.get('processing_rate', '0%'))
    
    console.print(table)


def single_query(question: str, config: Config):
    """单次查询"""
    console.print(f"\n[bold blue]🤔 问题:[/bold blue] {question}")
    
    try:
        from src import query as rag_query
        
        with console.status("[bold green]思考中..."):
            start_time = time.time()
            result = rag_query(question)
            end_time = time.time()
        
        # 显示回答
        console.print(f"\n[bold green]💡 回答:[/bold green]")
        console.print(result.get('answer', '抱歉，无法回答这个问题。'))
        
        # 显示元信息
        console.print(f"\n[dim]响应时间: {end_time - start_time:.2f}秒[/dim]")
        if 'confidence' in result:
            console.print(f"[dim]置信度: {result['confidence']:.2f}[/dim]")
        if 'sources' in result and result['sources']:
            console.print(f"[dim]参考文档: {len(result['sources'])} 个[/dim]")
            
    except Exception as e:
        console.print(f"[red]查询失败: {e}[/red]")


def interactive_mode(config: Config):
    """交互式查询模式"""
    console.print("\n[bold blue]🤖 进入交互式问答模式[/bold blue]")
    console.print("[dim]输入 'quit', 'exit' 或 'q' 退出[/dim]")
    
    try:
        from src import query_stream
        
        while True:
            # 获取用户输入
            question = Prompt.ask("\n[bold cyan]请输入您的问题[/bold cyan]")
            
            # 检查退出命令
            if question.lower() in ['quit', 'exit', 'q']:
                console.print("[yellow]再见！[/yellow]")
                break
            
            if not question.strip():
                continue
            
            # 流式查询
            console.print(f"\n[bold green]💡 回答:[/bold green]")
            try:
                for chunk in query_stream(question):
                    console.print(chunk, end='')
                console.print()  # 换行
            except Exception as e:
                console.print(f"[red]查询失败: {e}[/red]")
                
    except KeyboardInterrupt:
        console.print("\n[yellow]用户中断，退出程序[/yellow]")
    except Exception as e:
        console.print(f"[red]交互模式错误: {e}[/red]")


def import_documents(path: str, config: Config, db: Database, force: bool = False):
    """导入文档"""
    doc_path = Path(path)
    if not doc_path.exists():
        console.print(f"[red]路径不存在: {path}[/red]")
        return
    
    console.print(f"\n[bold blue]📥 导入文档[/bold blue]")
    console.print(f"路径: {doc_path.absolute()}")
    
    # 获取文档文件
    if doc_path.is_file():
        files = [doc_path]
    else:
        files = list(doc_path.glob("**/*.pdf")) + list(doc_path.glob("**/*.md"))
    
    if not files:
        console.print("[yellow]未找到支持的文档文件（PDF, MD）[/yellow]")
        return
    
    console.print(f"找到 {len(files)} 个文档文件")
    
    try:
        from src.document.processor import DocumentProcessor
        processor = DocumentProcessor(config, db, force_reprocess=force)
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("处理文档...", total=len(files))
            
            processed = 0
            for file_path in files:
                try:
                    progress.update(task, description=f"处理: {file_path.name}")
                    doc_ids = processor.process_file(str(file_path))
                    processed += 1
                    console.print(f"[green]✅ {file_path.name}: {len(doc_ids)} 个片段[/green]")
                except Exception as e:
                    console.print(f"[red]❌ {file_path.name}: {e}[/red]")
                
                progress.advance(task)
        
        console.print(f"\n[bold green]✅ 导入完成: {processed}/{len(files)} 个文件[/bold green]")
        
    except Exception as e:
        console.print(f"[red]导入失败: {e}[/red]")


@cli.command()
@click.pass_context
def init_db(ctx):
    """初始化数据库"""
    config = ctx.obj['config']
    db = ctx.obj['db']
    
    console.print("[bold blue]🗄️ 初始化数据库[/bold blue]")
    
    try:
        with console.status("[bold green]创建数据库表..."):
            db.create_tables()
        console.print("[bold green]✅ 数据库初始化完成[/bold green]")
    except Exception as e:
        console.print(f"[red]❌ 数据库初始化失败: {e}[/red]")


@cli.command()
@click.option('--days', default=30, help='保留天数')
@click.pass_context
def cleanup(ctx, days):
    """清理旧数据"""
    db = ctx.obj['db']
    
    console.print(f"[bold blue]🧹 清理 {days} 天前的数据[/bold blue]")
    
    try:
        with console.status("[bold green]清理中..."):
            deleted = db.cleanup_old_data(days)
        console.print(f"[bold green]✅ 清理完成: 删除 {deleted} 条记录[/bold green]")
    except Exception as e:
        console.print(f"[red]❌ 清理失败: {e}[/red]")


if __name__ == "__main__":
    cli() 
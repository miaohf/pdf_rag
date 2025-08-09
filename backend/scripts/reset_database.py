#!/usr/bin/env python3
"""
数据库重置脚本

清空数据库表并重新导入文档数据
"""

import sys
import traceback
from pathlib import Path
from sqlalchemy import text

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent))

from src.utils.config import Config
from src.utils.logger import get_logger
from src.core.database import Database
from src.core.models import Base
from src.document.processor import DocumentProcessor
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
import time

console = Console()
logger = get_logger(__name__)


def reset_database(config: Config, db: Database):
    """重置数据库"""
    console.print("[bold red]🗑️  清空数据库表...[/bold red]")
    
    try:
        with db.get_session() as session:
            # 删除所有表（按正确的依赖顺序）
            tables_to_drop = [
                "document_vectors",
                "document_chunks", 
                "documents",
                "query_history",
                "conversations"
            ]
            
            for table in tables_to_drop:
                try:
                    session.execute(text(f"DROP TABLE IF EXISTS {table} CASCADE"))
                    console.print(f"   ✅ 删除表 {table}")
                except Exception as e:
                    console.print(f"   ⚠️  删除表 {table} 失败: {e}")
            
            # 删除pgvector扩展（如果需要重新创建）
            try:
                session.execute(text("DROP EXTENSION IF EXISTS vector CASCADE"))
                console.print("   ✅ 删除 pgvector 扩展")
            except Exception as e:
                console.print(f"   ⚠️  删除 pgvector 扩展失败: {e}")
            
            session.commit()
            
    except Exception as e:
        logger.error(f"清空数据库失败: {e}")
        raise


def recreate_database(config: Config, db: Database):
    """重新创建数据库表"""
    console.print("[bold blue]🏗️  重新创建数据库表...[/bold blue]")
    
    try:
        with db.get_session() as session:
            # 重新创建pgvector扩展
            session.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            session.commit()
            console.print("   ✅ 创建 pgvector 扩展")
        
        # 使用ORM创建所有表
        Base.metadata.create_all(bind=db._engine)
        console.print("   ✅ 创建 ORM 表结构")
        
    except Exception as e:
        logger.error(f"重新创建数据库失败: {e}")
        raise


def import_all_documents(config: Config, db: Database):
    """重新导入所有文档"""
    console.print("[bold green]📄 重新导入文档...[/bold green]")
    
    docs_path = Path("docs")
    if not docs_path.exists():
        console.print("[red]❌ docs 目录不存在[/red]")
        return
    
    # 获取所有文档文件
    doc_files = []
    for ext in ["*.pdf", "*.md", "*.txt", "*.docx"]:
        doc_files.extend(docs_path.glob(ext))
    
    if not doc_files:
        console.print("[yellow]⚠️  docs 目录中没有找到文档文件[/yellow]")
        return
    
    console.print(f"   📊 找到 {len(doc_files)} 个文档文件")
    
    processor = DocumentProcessor(config, db)
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        
        for doc_file in doc_files:
            task = progress.add_task(f"处理: {doc_file.name}", total=None)
            
            try:
                success = processor.process_file(str(doc_file))
                if success:
                    console.print(f"   ✅ {doc_file.name}")
                else:
                    console.print(f"   ❌ {doc_file.name}: 处理失败")
            except Exception as e:
                console.print(f"   ❌ {doc_file.name}: {e}")
                logger.error(f"处理文件失败 {doc_file}: {e}")
            
            progress.remove_task(task)


def main():
    """主函数"""
    console.print("[bold cyan]🔄 开始重建数据库和重新导入文档[/bold cyan]")
    
    try:
        # 加载配置
        with console.status("[bold green]加载配置..."):
            config = Config()
        console.print("✅ 配置加载完成")
        
        # 连接数据库
        with console.status("[bold green]连接数据库..."):
            db = Database(config)
        console.print("✅ 数据库连接建立")
        
        # 重置数据库
        reset_database(config, db)
        
        # 重新创建表
        recreate_database(config, db)
        
        # # 重新导入文档
        import_all_documents(config, db)
        
        console.print("[bold green]🎉 数据库重置和文档导入完成！[/bold green]")
        
    except Exception as e:
        console.print(f"[bold red]❌ 操作失败: {e}[/bold red]")
        logger.error(f"数据库重置失败: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main() 
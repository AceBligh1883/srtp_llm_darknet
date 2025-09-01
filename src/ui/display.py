# src/ui/display.py
import os
import re
from typing import List
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.text import Text
from rich.table import Table
from src.search.rag_engine import RAGEngine
from src.common.logger import logger
from src.common.data_models import SearchResult
from src.ui.display_enhancements import display_image_in_terminal 

def _read_snippet_from_file(file_path: str, max_len: int = 300) -> tuple[str, bool]:
    """从文件路径安全地读取一个内容片段。"""
    if not os.path.exists(file_path):
        return "[内容文件不存在]", False
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read(max_len + 1)
        was_truncated = len(content) > max_len
        snippet = content[:max_len]
        return snippet, was_truncated
    except Exception as e:
        return f"[读取文件时出错: {e}]", False

def display_search_results(results: List[SearchResult], query: str):
    """使用 rich 库格式化并打印搜索结果"""
    console = Console()
    console.print(f"\n[bold green]与 “{query}” 相关的 {len(results)} 个结果[/bold green]\n")

    if not results:
        console.print(Panel("[yellow]未找到任何相关结果。[/yellow]", title="提示", border_style="yellow"))
        return
    
    for i, res in enumerate(results, 1):
        was_truncated = False
        if res.metadata.content_type == 'text':
            content_snippet, was_truncated = _read_snippet_from_file(res.metadata.file_path)
        else:
            content_snippet = f"图像文件，路径: {res.metadata.file_path}"

        highlighted_snippet = Text(content_snippet)
        if query and res.metadata.content_type == 'text':
            try:
                for match in re.finditer(re.escape(query), highlighted_snippet.plain, re.IGNORECASE):
                    highlighted_snippet.stylize("bold magenta on white", match.start(), match.end())
            except re.error:
                highlighted_snippet.highlight_words(query.split(), "bold magenta on white")

        meta_table = Table.grid(expand=True)
        meta_table.add_column(style="dim cyan", no_wrap=True)
        meta_table.add_column(ratio=1)
        meta_table.add_row("文件:", f" {os.path.basename(res.metadata.file_path)}")
        meta_table.add_row("类型:", f" {res.metadata.content_type}")
        
        panel_body = Table.grid(expand=True, padding=0)
        panel_body.add_column()
        panel_body.add_row(meta_table)
        panel_body.add_row("")
        panel_body.add_row(highlighted_snippet)

        if was_truncated:
            full_path_text = Text(f"\n... [查看完整内容]: {res.metadata.file_path}", style="dim")
            full_path_text.highlight_regex(r":\s.*$", "underline")
            panel_body.add_row(full_path_text)

        result_panel = Panel(
            panel_body,
            title=f"[bold cyan]#{i}[/bold cyan] [white]Score:[/white] [bold yellow]{res.score:.4f}[/bold yellow]",
            border_style="blue",
            expand=False,
            padding=(1, 2)
        )
        console.print(result_panel)
        console.print()

def render_rich_report(rag_engine, report_draft: str):
    """
    解析并渲染包含图像占位符的报告草稿。
    """
    report_table = Table.grid(padding=(0, 1, 1, 1))
    report_table.add_column()
    parts = re.split(r'(\[IMAGE: .+?\])', report_draft)
    for part in parts:
        if not part.strip():
            continue
        
        match = re.match(r'\[IMAGE: (image_\d+)\]', part)
        if match:
            placeholder_id = match.group(1)
            image_path = rag_engine.path_to_placeholder_map.get(placeholder_id)
            
            if image_path and os.path.exists(image_path):
                image_caption = f"[bold]引用资料:[/bold] [cyan]`{image_path}`[/cyan]"
                report_table.add_row(image_caption)
                display_image_in_terminal(image_path, max_width=80)
                report_table.add_row("")
            else:
                error_msg = f"[bold red]错误：报告引用了图片 '{placeholder_id}'，但未找到其路径 '{image_path}'。[/bold red]"
                report_table.add_row(error_msg)
        else:
            report_table.add_row(Markdown(part.strip()))
    return report_table

def display_rag_answer(question: str, answer_draft: str, rag_engine: RAGEngine, image_path: str = None):
    """
    打印 RAG 问答结果。
    """
    console = Console()
    console.print()

    if image_path:
        query_display = f"[bold]Image Query:[/bold] {image_path}\n[bold]Q:[/bold] {question or '无附加问题'}"
    else:
        query_display = f"[bold]Q:[/bold] {question}"
    
    question_panel = Panel(
        query_display,
        title="[bold yellow]您的查询[/bold yellow]",
        border_style="yellow",
        expand=False
    )
    console.print(question_panel)
    
    report_content_table = render_rich_report(rag_engine, answer_draft)

    answer_panel = Panel(
        report_content_table,
        title="[bold green]AI的回答[/bold green]",
        border_style="green",
        expand=False
    )
    console.print(answer_panel)
    console.print()

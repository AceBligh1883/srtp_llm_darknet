# src/ui/presenter.py
import os
import re
from typing import List
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.table import Table
from rich.markdown import Markdown
from src.common.data_models import RAGReport, SearchResult

class Presenter:
    """
    负责将数据对象（如 RAGReport）渲染为终端富文本输出的专职类。
    """
    def __init__(self):
        self.console = Console()

    def display_rag_report(self, report: RAGReport):
        """
        接收一个 RAGReport 对象并将其完整渲染。
        """
        self.console.print()

        query_panel = self._render_query_panel(report)
        self.console.print(query_panel)
        
        answer_panel = self._render_answer_panel(report)
        self.console.print(answer_panel)
        self.console.print()

    def _render_query_panel(self, report: RAGReport) -> Panel:
        """创建并返回查询信息面板。"""
        query_text = report.question if report.question else "对图片进行综合分析。"
        
        query_display = Text.from_markup(f"[bold]Q:[/bold] {query_text}")
        if report.query_image_path:
            image_line = Text.from_markup(f"\n[bold]Image Query:[/bold] [cyan]`{report.query_image_path}`[/cyan]")
            query_display.append_text(image_line)
        
        return Panel(query_display, title="[bold yellow]您的查询[/bold yellow]", border_style="yellow")
        
    def _render_answer_panel(self, report: RAGReport) -> Panel:
        """创建并返回包含AI回答和附录的主面板。"""
        report_content = Markdown(report.answer, style="default")
        appendix_content = self._render_appendix(report.evidence)
        
        full_report_table = Table.grid(expand=True)
        full_report_table.add_row(report_content)
        full_report_table.add_row(appendix_content)

        return Panel(full_report_table, title="[bold green]AI的回答[/bold green]", border_style="green")
    
    def _render_appendix(self, evidence: List[SearchResult]) -> Panel:
        """创建并返回证据附录面板。"""
        if not evidence:
            return Panel("[dim]报告未引用任何外部证据。[/dim]", title="附录", border_style="dim")

        text_lines = [f"  {i+1}. 文件: `{r.metadata.file_path}` (相关性: {r.score:.4f})" for i, r in enumerate(r for r in evidence if r.metadata.content_type == 'text')]
        image_lines = [f"  {i+1}. 文件: `{r.metadata.file_path}` (相关性: {r.score:.4f})" for i, r in enumerate(r for r in evidence if r.metadata.content_type == 'image')]
        
        text_content = "\n".join(text_lines) or "  [dim]无[/dim]"
        image_content = "\n".join(image_lines) or "  [dim]无[/dim]"

        appendix_content = (
            f"[bold]引用的文本证据:[/bold]\n{text_content}\n\n"
            f"[bold]参考的视觉资料:[/bold]\n{image_content}"
        )

        return Panel(appendix_content, title="证据与资料来源附录", border_style="blue")
    
    def display_search_results(self, results: List[SearchResult], query: str):
        """使用 rich 库格式化并打印搜索结果"""
        self.console.print(f"\n[bold green]与 “{query}” 相关的 {len(results)} 个结果[/bold green]\n")

        if not results:
            self.console.print(Panel("[yellow]未找到任何相关结果。[/yellow]", title="提示", border_style="yellow"))
            return
        
        for i, res in enumerate(results, 1):
            result_panel = self._render_single_search_result(res, query, i)
            self.console.print(result_panel)
            self.console.print()

    def _render_single_search_result(self, res: SearchResult, query: str, index: int) -> Panel:
        """渲染单个搜索结果的面板。"""
        was_truncated = False
        if res.metadata.content_type == 'text':
            content_snippet, was_truncated = self._read_snippet_from_file(res.metadata.file_path)
        else:
            content_snippet = f"图像文件，路径: {res.metadata.file_path}"

        highlighted_snippet = Text(content_snippet)
        if query and res.metadata.content_type == 'text':
            highlighted_snippet.highlight_regex(re.escape(query), "bold magenta on white")

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

        return Panel(
            panel_body,
            title=f"[bold cyan]#{index}[/bold cyan] [white]Score:[/white] [bold yellow]{res.score:.4f}[/bold yellow]",
            border_style="blue",
            expand=False,
            padding=(1, 2)
        )

    def _read_snippet_from_file(self, file_path: str, max_len: int = 300) -> tuple[str, bool]:
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
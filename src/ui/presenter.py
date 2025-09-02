# src/ui/presenter.py
import os
import re
from typing import List, Dict
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.table import Table
from src.common.data_models import RAGReport, SearchResult
from src.ui.display_enhancements import display_image_in_terminal

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

        query_display = f"[bold]Q:[/bold] {report.question or '无附加问题'}"
        if report.query_image_path:
            query_display = f"[bold]Image Query:[/bold] {report.query_image_path}\n{query_display}"
        
        question_panel = Panel(query_display, title="[bold yellow]您的查询[/bold yellow]", border_style="yellow")
        self.console.print(question_panel)
        
        report_content = self._render_report_body(report.body, report.image_references)
        appendix_content = self._render_appendix(report.evidence)
        
        full_report_content = Table.grid(expand=True)
        full_report_content.add_row(report_content)
        full_report_content.add_row(appendix_content)

        answer_panel = Panel(full_report_content, title="[bold green]AI的回答[/bold green]", border_style="green")
        self.console.print(answer_panel)
        self.console.print()

    def _render_report_body(self, body: str, image_references: Dict[str, str]) -> Table:
        """
        解析和显示内联图片。
        """
        body_table = Table.grid(padding=(0, 1, 1, 1), expand=True)
        body_table.add_column()
        parts = re.split(r'(\[IMAGE: .+?\])', body)
        
        for part in parts:
            if not part.strip(): continue
            
            match = re.match(r'\[IMAGE: (image_\d+|main_query_image)\]', part)
            if match:
                placeholder_id = match.group(1)
                image_path = image_references.get(placeholder_id)
                if image_path and os.path.exists(image_path):
                    image_caption = f"[bold]引用资料:[/bold] [cyan]`{image_path}`[/cyan]"
                    body_table.add_row(image_caption)
                    display_image_in_terminal(image_path, max_width=80)
                    body_table.add_row("")
                else:
                    error_msg = f"[bold red]错误：报告引用了图片 '{placeholder_id}'，但未找到其路径。[/bold red]"
                    body_table.add_row(error_msg)
            else:
                body_table.add_row(Markdown(part.strip()))
        return body_table
        
    def _render_appendix(self, evidence: List[SearchResult]) -> Panel:
        """
        使用从 RAGReport 接收到的证据来渲染附录。
        """
        if not evidence:
            return Panel("[dim]报告未引用任何外部证据。[/dim]", title="附录", border_style="dim")

        text_lines = [f"  {i+1}. 文件: `{r.metadata.file_path}` (相关性: {r.score:.4f})" for i, r in enumerate(r for r in evidence if r.metadata.content_type == 'text')]
        image_lines = [f"  {i+1}. 文件: `{r.metadata.file_path}` (相关性: {r.score:.4f})" for i, r in enumerate(r for r in evidence if r.metadata.content_type == 'image')]
        
        appendix_table = Table.grid(padding=(0, 1), expand=True)
        appendix_table.add_column()
        appendix_table.add_row("[bold]引用的文本证据:[/bold]")
        appendix_table.add_row(Markdown("\n".join(text_lines) or "  [dim]无[/dim]"))
        appendix_table.add_row("")
        appendix_table.add_row("[bold]参考的视觉资料:[/bold]")
        appendix_table.add_row(Markdown("\n".join(image_lines) or "  [dim]无[/dim]"))
            
        return Panel(appendix_table, title="证据与资料来源附录", border_style="blue")

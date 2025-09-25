# src/ui/kg_presenter.py
from rich.console import Console
from rich.tree import Tree
from typing import List, Dict

class KGPresenter:
    """
    负责在知识图谱构建过程中，以美观的方式展示新提取的三元组。
    """
    def __init__(self):
        self.console = Console()

    def display_new_triples(self, doc_id: str, triples: List[Dict]):
        """
        为单个文档创建一个可折叠的树状图来展示新入库的三元组。

        Args:
            doc_id (str): 文档的ID。
            triples (List[Dict]): 从该文档中提取的三元组列表。
        """
        if not triples:
            return

        tree = Tree(f"[bold green]新知识入库: {doc_id}[/bold green] ({len(triples)}个三元组)")
        
        for t in triples:
            head_type = t.get('head_type', 'UNKNOWN')
            head_name = t.get('head', 'N/A')
            relation = t.get('relation', 'RELATED_TO')
            tail_type = t.get('tail_type', 'UNKNOWN')
            tail_name = t.get('tail', 'N/A')

            head_str = f"([cyan]{head_type}[/cyan]) [bright_cyan]{head_name}[/bright_cyan]"
            relation_str = f"-[ [yellow]{relation}[/yellow] ]->"
            tail_str = f"([magenta]{tail_type}[/magenta]) [bright_magenta]{tail_name}[/bright_magenta]"
            
            tree.add(f"{head_str} {relation_str} {tail_str}")
            
        self.console.print(tree)


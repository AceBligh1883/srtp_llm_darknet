# src/ui/display_enhancements.py

import os
from term_image.image import from_file
from src.common.logger import logger
from rich.console import Console

console = Console()

def display_image_in_terminal(image_path: str, max_width: int = 100):
    """
    尝试在兼容的终端（如VS Code）中直接显示图片。
    如果终端不支持或文件不存在，则不执行任何操作。

    Args:
        image_path (str): 要显示的图片的路径。
        max_width (int): 图片在终端中显示的最大宽度（字符数）。
    """
    if not os.path.exists(image_path):
        logger.warning(f"尝试显示图片失败：文件不存在 {image_path}")
        return

    try:
        image = from_file(image_path, width=max_width)
        image.draw()
    except Exception as e:
        logger.error(f"在终端中显示图片 {image_path} 时出错: {e}")
        console.print(f"[italic yellow]（无法预览图片: {os.path.basename(image_path)}）[/italic yellow]")


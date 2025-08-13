# src/features/text_processing.py
"""
简化的、目标明确的文本预处理器
"""
import re
import unicodedata

class TextProcessor:
    """为模型准备文本"""

    def __init__(self):
        self.html_tags_pattern = re.compile(r'<[^>]+>')
        self.extra_whitespace_pattern = re.compile(r'\s+')

    def process(self, text: str) -> str:
        """
        执行简化的文本预处理流程。
        1. 清理HTML标签
        2. Unicode规范化 (NFKC)
        3. 统一空白符
        4. 转换为小写
        """
        if not text:
            return ""

        text = self.html_tags_pattern.sub(' ', text)
        text = unicodedata.normalize('NFKC', text)
        text = self.extra_whitespace_pattern.sub(' ', text).strip()
        
        return text.lower()

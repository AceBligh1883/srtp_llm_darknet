# src/processing/text_processor.py
import re
import unicodedata

class TextProcessor:
    """为模型准备文本"""
    def __init__(self):
        self.html_tags_pattern = re.compile(r'<.*?>') 
        self.extra_whitespace_pattern = re.compile(r'\s+')

    def process(self, text: str) -> str:
        if not text: return ""
        
        text = self.html_tags_pattern.sub(' ', text)
        text = unicodedata.normalize('NFKC', text)
        text = self.extra_whitespace_pattern.sub(' ', text).strip()
        
        return text.lower()

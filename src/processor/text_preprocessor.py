# -*- coding: utf-8 -*-
"""
文本预处理模块
"""

import re
import nltk
import unicodedata
from typing import Set
from src.logger import logger
from src.utils.text_splitter import TextSplitter

class TextPreprocessor:
    """
    文本预处理模块
    主要功能：
    1. 移除HTML标签;
    2. 移除表情符号;
    3. 规范化文本;
    4. 移除基于NLTK的英文停用词;
    """
    def __init__(self, max_text_length: int = 1024) -> None:
        self._init_nltk_resources()
        self._init_patterns()
        self._init_stopwords()
        self.max_text_length = max_text_length
        self.text_chunker = TextSplitter()
        
    def _init_nltk_resources(self) -> None:
        """初始化NLTK资源"""
        try:
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('corpora/stopwords')
        except LookupError:
            logger.info("下载NLTK资源...")
            nltk.download(['punkt', 'stopwords'])

    def _init_patterns(self) -> None:
        """初始化正则表达式模式"""
        self.cleanup_patterns = {
            'EXTRA_WHITESPACE': re.compile(r'\s+'),
            'HTML_TAGS': re.compile(r'<[^>]+>'),
            'EMOJI': re.compile(r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF]')
        }

    def _init_stopwords(self) -> None:
        """初始化停用词"""
        from nltk.corpus import stopwords
        self.stop_words: Set[str] = set(stopwords.words('english'))

    def normalize_text(self, text: str) -> str:
        """
        文本规范化处理
        - Unicode NFKC标准化
        - Unicode NFKC标准化
        - 统一空白字符
        """
        text = unicodedata.normalize('NFKC', text)
        text = re.sub(r'([!?,.]){2,}', r'\1', text)
        text = self.cleanup_patterns['EXTRA_WHITESPACE'].sub(' ', text)
        
        return text.strip()

    def clean_text(self, text: str) -> str:
        """
        文本清理
        """
        text = self.cleanup_patterns['HTML_TAGS'].sub('', text)
        text = self.cleanup_patterns['EMOJI'].sub('', text)
        return text

    def remove_stopwords(self, text: str) -> str:
        """移除停用词"""
        words = nltk.word_tokenize(text)
        return ' '.join(word for word in words 
                       if word.lower() not in self.stop_words)

    def preprocess(self, text: str) -> str:
        """
        执行完整的文本预处理流程
        处理步骤:
        1. 文本规范化
        2. 基础清理
        3. 停用词处理
        4. 长度控制
        """
        if not text:
            return ""
            
        text = self.normalize_text(text)
        text = self.clean_text(text)
        text = self.remove_stopwords(text)
        if len(text) > self.max_text_length:
            text = self.text_chunker.extract_key_sections(text, self.max_text_length)
            
        return text.lower()
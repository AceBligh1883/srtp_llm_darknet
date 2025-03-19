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
    1. 对文本中的敏感信息进行脱敏处理；
    2. 去除文本中所有标点符号；
    3. 替换暗网常见黑话为对应同义词；
    4. 移除停用词。
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
        # 敏感信息模式
        self.patterns = {
            'CRYPTO_ADDR': re.compile(r'\b(bc1|[13])[a-zA-HJ-NP-Z0-9]{25,39}\b'),
            'PHONE': re.compile(r'[\+\(]?[1-9][0-9 .\-\(\)]{8,}[0-9]'),
            'EMAIL': re.compile(r'\b[\w\.-]+@[\w\.-]+\.\w+\b'),
            'IP_ADDR': re.compile(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b'),
            'URL': re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'),
            'CREDIT_CARD': re.compile(r'\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b'),
            'SSN': re.compile(r'\b\d{3}-\d{2}-\d{4}\b')
        }

        # 清理模式
        self.cleanup_patterns = {
            'EXTRA_WHITESPACE': re.compile(r'\s+'),
            'SPECIAL_CHARS': re.compile(r'[^\w\s]'),
            'NUMBERS': re.compile(r'\d+'),
            'HTML_TAGS': re.compile(r'<[^>]+>'),
            'URLS': re.compile(r'http\S+|www\S+'),
            'EMOJI': re.compile(r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF]')
        }

    def _init_stopwords(self) -> None:
        """初始化停用词"""
        from nltk.corpus import stopwords
        self.stop_words: Set[str] = set(stopwords.words('english'))
        
        # 添加自定义停用词
        custom_stopwords = {
            "login", "register", "signup", "signin", "logout",
            "username", "password", "email", "admin", "user",
            "cart", "checkout", "payment", "shipping", "order",
            "price", "discount", "sale", "buy", "sell",
            "like", "share", "comment", "follow", "post",
            "tweet", "message", "friend", "profile", "status",
            "home", "about", "contact", "search", "help",
            "faq", "terms", "privacy", "policy", "menu"
        }
        self.stop_words.update(custom_stopwords)

    def normalize_text(self, text: str) -> str:
        """
        文本规范化处理
        - Unicode标准化
        - 去除重复标点
        - 统一空白字符
        """
        text = unicodedata.normalize('NFKC', text)
        text = re.sub(r'([!?,.]){2,}', r'\1', text)
        text = self.cleanup_patterns['EXTRA_WHITESPACE'].sub(' ', text)
        
        return text.strip()

    def clean_text(self, text: str) -> str:
        """
        文本清理
        - 移除HTML标签
        - 移除URL
        - 移除表情符号
        - 替换数字
        """
        text = self.cleanup_patterns['HTML_TAGS'].sub('', text)
        text = self.cleanup_patterns['URLS'].sub('[URL]', text)
        text = self.cleanup_patterns['EMOJI'].sub('', text)
        text = self.cleanup_patterns['NUMBERS'].sub('[NUM]', text)
        return text
    
    def anonymize(self, text: str) -> str:
        """敏感信息脱敏"""
        for tag, pattern in self.patterns.items():
            text = pattern.sub(f'[{tag}]', text)
        return text

    def remove_stopwords(self, text: str) -> str:
        """移除停用词"""
        words = nltk.word_tokenize(text)
        return ' '.join(word for word in words 
                       if word.lower() not in self.stop_words)

    def preprocess(self, text: str) -> str:
        """
        完整的预处理流程
        1. 文本规范化
        2. 文本清理
        3. 敏感信息脱敏
        4. 移除停用词
        5. 长度控制
        """
        if not text:
            return ""
            
        text = self.normalize_text(text)
        text = self.clean_text(text)
        text = self.anonymize(text)
        text = self.remove_stopwords(text)
        if len(text) > self.max_text_length:
            text = self.text_chunker.extract_key_sections(text, self.max_text_length)
            
        return text.lower()
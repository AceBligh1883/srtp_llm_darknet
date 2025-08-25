# src/features/translator.py

import translators as ts
from src.common.logger import logger

def translate_to_english(query_text: str) -> str:
    """
    将文本翻译成英文。
    
    Args:
        query_text (str): 用户输入的查询文本。

    Returns:
        str: 翻译后的英文文本或原始文本。
    """
    try:
        query_text.encode('ascii')
        logger.debug(f"查询 '{query_text}' 已是英文，跳过翻译。")
        return query_text
    except UnicodeEncodeError:
        pass

    try:
        translated_text = ts.translate_text(query_text, translator='google', to_language='en')
        logger.info(f"成功将查询 '{query_text}' 翻译为 '{translated_text}'")
        return translated_text
    except Exception as e:
        logger.warning(f"翻译查询 '{query_text}' 失败: {e}。将使用原始查询。")
        return query_text


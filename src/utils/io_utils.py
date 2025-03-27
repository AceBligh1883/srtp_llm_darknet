# -*- coding: utf-8 -*-
"""
文件处理工具模块
"""

import os
import json
import aiofiles
from typing import List, Dict, Any
from datetime import datetime
from urllib.parse import urlparse

from src.logger import logger
from config import config
from src.analyzer.content_analyzer import DarknetAnalyzer

def load_texts(directory: str = config.TEXT_DIR, min_length: int = config.MIN_TEXT_LENGTH) -> List[Dict[str, str]]:
    """
    从指定目录加载所有文本文件内容。
    
    :param directory: 文件目录路径
    :param min_length: 最小文本长度
    :return: 包含文件名和内容的字典列表
    """
    data: List[Dict[str, str]] = []
    skipped_files = 0
    
    if not os.path.exists(directory):
        logger.error(f"目录不存在：{directory}")
        return data
    
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            file_path = os.path.join(directory, filename)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # 过滤过短的文本
                if len(content.encode('utf-8')) < min_length:
                    logger.debug(f"跳过过短文本: {filename} ({len(content.encode('utf-8'))}字节)")
                    skipped_files += 1
                    continue
                    
                data.append({"filename": filename, "content": content})
            except Exception as e:
                logger.error(f"文件读取失败：{filename} - {str(e)}")
    
    if skipped_files > 0:
        logger.info(f"已跳过 {skipped_files} 个过短文本文件")
    
    return data

def save_report(report: Dict[str, Any], filename: str, output_dir: str = config.OUTPUT_DIR) -> None:
    """
    将词向量保存为 JSON 文件。
    
    :param report: 包含词向量的字典
    :param filename: 文件名
    :param output_dir: 输出目录路径
    """
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}.json")
    
    # 只保存词向量数据
    vector_data = {
        "embedding": report.get("embedding")
    }
    
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(vector_data, f, ensure_ascii=False)
        logger.info(f"向量数据已保存: {output_path}")
    except Exception as e:
        logger.error(f"向量数据保存失败：{output_path} - {str(e)}")

def process_file(file_item: Dict[str, str], analyzer: DarknetAnalyzer, 
                output_dir: str = config.OUTPUT_DIR, process_image: bool = True) -> bool:
    """
    处理单个文件：生成文本和图像的词向量。
    """
    filename = file_item["filename"]
    content = file_item["content"]
    
    base_name = os.path.splitext(filename)[0]
    image_path = None
    
    if process_image:
        for ext in config.SUPPORTED_IMAGE_FORMATS:
            temp_path = os.path.join(config.IMAGE_DIR, f"{base_name}{ext}")
            if os.path.exists(temp_path):
                image_path = temp_path
                break
    
    logger.info(f"开始处理文件：{filename}")
    
    vectors = analyzer.analyze(content, image_path)
    save_report(vectors, filename, output_dir)

    return True

async def save_content(content, url, content_type):
    """
    保存各种类型的内容到对应目录
    
    :param content: 要保存的内容
    :param url: 内容来源URL
    :param content_type: 内容类型 (text, screenshot, image, video, file)
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    parsed = urlparse(url)
    base = f"{timestamp}_{parsed.netloc}_{hash(url) % 10000}"
    
    if content_type == "text":
        filename = f"{base}.txt"
        path = os.path.join(config.TEXT_DIR, filename)
        try:
            async with aiofiles.open(path, "w", encoding="utf-8") as f:
                await f.write(content)
            logger.info(f"[文本] 已保存至: {path}")
        except Exception as e:
            logger.error(f"保存文本失败: {e}")
    elif content_type in ["screenshot", "image", "video", "file"]:
        ext_map = {
            "screenshot": ".png",
            "image": ".jpg",
            "video": ".mp4",
            "file": ".bin"
        }
        ext = ext_map.get(content_type, ".bin")
        dir_map = {
            "screenshot": config.SCREENSHOTS_DIR,
            "image": config.IMAGE_DIR,
            "video": config.VIDEOS_DIR,
            "file": config.FILES_DIR
        }
        filename = f"{base}{ext}"
        path = os.path.join(dir_map[content_type], filename)
        try:
            async with aiofiles.open(path, "wb") as f:
                await f.write(content)
            logger.info(f"[{content_type.upper()}] 已保存至: {path}")
        except Exception as e:
            logger.error(f"保存{content_type}失败: {e}")

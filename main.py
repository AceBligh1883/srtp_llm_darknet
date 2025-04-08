#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
暗网文本和图像智能分析系统
"""

import os
import sys
import time
import argparse
import asyncio

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.logger import logger
from config import config
from src.analyzer.content_analyzer import DarknetAnalyzer
from src.processor.text_processor import TextProcessor
from src.processor.image_processor import ImageProcessor
from src.crawler.coordinator import Coordinator
from src.crawler.worker import Worker

def setup_dirs() -> None:
    """确保必要的目录存在"""
    directories = [
        config.TEXT_DIR, 
        config.OUTPUT_DIR, 
        config.IMAGE_DIR,
        config.SCREENSHOTS_DIR,
        config.VIDEOS_DIR,
        config.FILE_DIR
    ]
    
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            logger.info(f"创建目录：{directory}")

def parse_args() -> argparse.Namespace:
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='暗网文本和图像智能分析系统')
    
    # 创建互斥的主要操作组
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument('--text', action='store_true', help='仅处理文本文件')
    mode_group.add_argument('--images', action='store_true', help='仅处理图像文件')
    mode_group.add_argument('--all', action='store_true', help='同时处理文本和图像')
    mode_group.add_argument('--crawl', action='store_true', help='启动爬虫')
    mode_group.add_argument('--init-queue', action='store_true', help='初始化爬虫任务队列')
    mode_group.add_argument('--queue-status', action='store_true', help='查看爬虫队列状态')
    
    # 爬虫相关参数
    crawler_group = parser.add_argument_group('爬虫选项')
    crawler_group.add_argument('--url', help='添加自定义种子URL')
    crawler_group.add_argument('--clear', action='store_true', help='清空现有队列')
    
    return parser.parse_args()

async def run_crawler():
    """运行爬虫"""
    worker = Worker()
    await worker.run()

def main() -> None:
    """主程序入口"""
    start_time = time.time()
    logger.info("暗网智能分析系统启动")
    
    try:
        setup_dirs()
        args = parse_args()
        
        # 爬虫相关操作
        if args.crawl or args.init_queue or args.queue_status:
            coordinator = Coordinator()
            
            if args.init_queue:
                seed_urls = None
                if args.url:
                    seed_urls = [args.url]
                if coordinator.initialize_queue(seed_urls, args.clear):
                    logger.info("任务队列初始化完成")
            
            elif args.queue_status:
                status = coordinator.get_queue_status()
                logger.info(f"队列状态: 待处理任务 {status['queue_size']}，已访问URL {status['visited_size']}")
            
            elif args.crawl:
                asyncio.run(run_crawler())
        
        # 分析器相关操作
        else:
            # 初始化分析器
            analyzer = DarknetAnalyzer(config.MAX_TEXT_LENGTH)
            
            # 初始化处理器
            text_processor = TextProcessor(analyzer)
            image_processor = ImageProcessor(analyzer)
            
            # 根据参数执行相应的处理
            if args.text or args.all:
                text_processor.process()
                
            if args.images or args.all:
                image_processor.process()
            
    except Exception as e:
        logger.error(f"处理过程中发生错误: {str(e)}")
    finally:
        elapsed_time = time.time() - start_time
        logger.info(f"处理完成，总耗时 {elapsed_time:.2f} 秒")

if __name__ == "__main__":
    main()

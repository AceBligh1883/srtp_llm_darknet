# src/main.py
import os
import time
import argparse

import asyncio
from src.crawler.coordinator import Coordinator
from src.crawler.worker import Worker

from src.common import config
from src.common.logger import logger
from src.pipeline.processing import ProcessingPipeline
from src.search.engine import SearchEngine


def setup_dirs():
    """确保所有必要的输入数据目录存在"""
    dirs = [config.TEXT_DIR, config.IMAGE_DIR]
    for d in dirs:
        if not os.path.exists(d):
            os.makedirs(d)
            logger.info(f"创建目录: {d}")

def handle_crawler(args: argparse.Namespace):
    """处理爬虫相关命令"""
    if not (args.crawl_init or args.crawl_status or args.crawl):
        return

    coordinator = Coordinator()
    if args.crawl_init:
        coordinator.initialize_queue(clear_existing=args.reset_queue)
        logger.info("爬虫队列初始化完成。")
    
    elif args.crawl_status:
        status = coordinator.get_queue_status()
        logger.info(f"爬虫队列状态: 待处理任务 {status['queue_size']}，已访问URL {status['visited_size']}")

    elif args.crawl:
        logger.info("启动爬虫...")
        worker = Worker(use_screenshot=args.screenshot)
        asyncio.run(worker.run())

def handle_process(args: argparse.Namespace):
    """处理内容索引命令"""
    if not (args.process_text or args.process_image or args.process_all):
        return

    pipeline = ProcessingPipeline()
    if args.process_text or args.process_all:
        pipeline.process_directory(config.TEXT_DIR, 'text')
    if args.process_image or args.process_all:
        pipeline.process_directory(config.IMAGE_DIR, 'image')

def handle_search(args: argparse.Namespace):
    """处理内容检索命令"""
    if not (args.search_text or args.search_image):
        return

    engine = SearchEngine()
    if args.search_text:
        results = engine.search_by_text(args.search_text, args.limit)
    elif args.search_image:
        results = engine.search_by_image(args.search_image, args.limit)
    else:
        return

    logger.info(f"找到 {len(results)} 个相似结果:")
    if not results:
        return
        
    for i, res in enumerate(results, 1):
        print("-" * 20)
        print(f"{i}. Score: {res.score:.4f}")
        print(f"   ID: {res.doc_id}")
        print(f"   Type: {res.metadata.content_type}")
        print(f"   Path: {res.metadata.file_path}")
        print(f"   Timestamp: {res.metadata.timestamp}")

def main():
    parser = argparse.ArgumentParser(description="暗网多模态分析与检索系统")


    crawler_group = parser.add_argument_group('爬虫命令')
    crawler_group.add_argument('--crawl-init', action='store_true', help='使用配置文件中的种子URL初始化任务队列')
    crawler_group.add_argument('--reset-queue', action='store_true', help='在初始化时清空现有队列')
    crawler_group.add_argument('--crawl-status', action='store_true', help='查看当前任务队列状态')
    crawler_group.add_argument('--crawl', action='store_true', help='启动爬虫工作器开始爬取')
    crawler_group.add_argument('--screenshot', action='store_true', help='在爬取时启用截图功能（需要配置好Tor浏览器和驱动）')

    parser.add_argument('--limit', type=int, default=10, help='返回结果数量')
    process_group = parser.add_argument_group('内容处理命令')
    process_group.add_argument('--process-text', action='store_true', help='处理并索引所有文本文件')
    process_group.add_argument('--process-image', action='store_true', help='处理并索引所有图像文件')
    process_group.add_argument('--process-all', action='store_true', help='处理并索引所有内容')

    search_group = parser.add_argument_group('内容检索命令')
    search_group.add_argument('--search-text', metavar='QUERY', help='根据文本查询相似内容')
    search_group.add_argument('--search-image', metavar='IMAGE_PATH', help='根据图像查询相似内容')
    
    args = parser.parse_args()
    
    if not any(vars(args).values()):
        parser.print_help()
        return

    start_time = time.time()
    logger.info("系统启动...")
    
    setup_dirs()
    handle_crawler(args)
    handle_process(args)
    handle_search(args)

    elapsed = time.time() - start_time
    logger.info(f"操作完成，总耗时: {elapsed:.2f} 秒")

if __name__ == "__main__":
    main()

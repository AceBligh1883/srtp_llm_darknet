# main.py
import os
import time
import argparse
import asyncio
from src.crawler.coordinator import Coordinator
from src.crawler.worker import Worker
from src.common import config
from src.common.logger import logger
from src.pipeline.processing import ProcessingPipeline
from src.pipeline.kg_pipeline import run_kg_construction_pipeline
from src.search.engine import SearchEngine
from src.search.rag_engine import RAGEngine
from src.ui.display import display_search_results, display_rag_answer

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
        for image_dir in config.IMAGE_SOURCE_DIRS:
            pipeline.process_directory(image_dir, 'image')

def handle_search(args: argparse.Namespace):
    """处理内容检索命令"""
    query_term = args.search_text or args.search_image
    if not query_term:
        return

    engine = SearchEngine()
    if args.search_text:
        results = engine.search_by_text(args.search_text, args.limit)
    elif args.search_image:
        results = engine.search_by_image(args.search_image, args.limit)

    display_search_results(results, query_term)

def handle_rag(args: argparse.Namespace):
    """处理RAG问答命令"""
    if not (args.ask or args.query_image):
        return

    rag_engine = RAGEngine()
    question = args.ask
    image_path = args.query_image
    answer = ""

    from rich.console import Console 
    with Console().status("[bold cyan]AI正在思考中...[/bold cyan]", spinner="dots"):
        if image_path:
            answer = rag_engine.ask_with_image(image_path, question)
        elif question:
            answer = rag_engine.ask(question)
    
    display_rag_answer(question, answer, image_path)

def handle_kg_build(args: argparse.Namespace):
    """处理知识图谱构建命令。"""
    if not args.build_kg:
        return
    run_kg_construction_pipeline()

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
    
    rag_group = parser.add_argument_group('RAG问答命令')
    rag_group.add_argument('--ask', type=str, help='向RAG引擎提问。可与 --query-image 组合使用，对图像进行提问。')
    rag_group.add_argument('--query-image', type=str, help='提供图像文件路径作为RAG的主要上下文。')

    kg_group = parser.add_argument_group('知识图谱命令')
    kg_group.add_argument('--build-kg', action='store_true', help='从所有文本文档中提取知识并构建图谱')

    args = parser.parse_args()
    
    if not any(vars(args).values()):
        parser.print_help()
        return

    start_time = time.time()
    logger.debug("系统启动...")
    
    setup_dirs()
    handle_crawler(args)
    handle_process(args)
    handle_search(args)
    handle_rag(args)
    handle_kg_build(args)
    
    elapsed = time.time() - start_time
    logger.info(f"操作完成，总耗时: {elapsed:.2f} 秒")

if __name__ == "__main__":
    main()

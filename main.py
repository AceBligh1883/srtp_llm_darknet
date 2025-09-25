# main.py
import os
import time
import argparse
import asyncio
from src.crawler.coordinator import Coordinator
from src.crawler.worker import Worker
from src.common import config
from src.common.logger import logger
from src.pipeline.indexing_pipeline import IndexingPipeline 
from src.pipeline.kg_pipeline import run_kg_construction_pipeline
from src.search.engine import SearchEngine
from src.search.rag_engine import RAGEngine
from src.ui.presenter import Presenter 

def setup_dirs():
    """确保所有必要的输入数据目录存在"""
    dirs = [config.TEXT_DIR, config.IMAGE_DIR]
    for d in dirs:
        if not os.path.exists(d):
            os.makedirs(d)
            logger.info(f"创建目录: {d}")

def main():
    parser = argparse.ArgumentParser(description="暗网多模态分析与检索系统")
    parser.add_argument('--limit', type=int, default=10, help='返回结果数量')

    # --- 定义所有命令组 ---
    crawler_group = parser.add_argument_group('爬虫命令')
    crawler_group.add_argument('--crawl-init', action='store_true', help='使用配置文件中的种子URL初始化任务队列')
    crawler_group.add_argument('--reset-queue', action='store_true', help='在初始化时清空现有队列')
    crawler_group.add_argument('--crawl-status', action='store_true', help='查看当前任务队列状态')
    crawler_group.add_argument('--crawl', action='store_true', help='启动爬虫工作器开始爬取')
    crawler_group.add_argument('--screenshot', action='store_true', help='在爬取时启用截图功能')

    process_group = parser.add_argument_group('内容处理命令')
    process_group.add_argument('--process-text', action='store_true', help='处理并索引所有文本文件')
    process_group.add_argument('--process-image', action='store_true', help='处理并索引所有图像文件')
    process_group.add_argument('--process-all', action='store_true', help='处理并索引所有内容')

    search_group = parser.add_argument_group('内容检索命令')
    search_group.add_argument('--search-text', metavar='QUERY', help='根据文本查询相似内容')
    search_group.add_argument('--search-image', metavar='IMAGE_PATH', help='根据图像查询相似内容')
    
    rag_group = parser.add_argument_group('RAG问答命令')
    rag_group.add_argument('--ask', type=str, help='向RAG引擎提问。可与 --query-image 组合使用。')
    rag_group.add_argument('--query-image', type=str, help='提供图像文件路径作为RAG的上下文。')

    kg_group = parser.add_argument_group('知识图谱命令')
    kg_group.add_argument('--build-kg', action='store_true', help='从所有文本文档中提取知识并构建图谱')
    kg_group.add_argument('--workers', type=int, default=1, help='知识图谱构建时使用的并发工作线程数。')

    args = parser.parse_args()
    
    action_taken = any([
        args.crawl_init, args.crawl_status, args.crawl,
        args.process_text, args.process_image, args.process_all,
        args.search_text, args.search_image,
        args.ask, args.query_image,
        args.build_kg
    ])
    if not action_taken:
        parser.print_help()
        return

    start_time = time.time()
    logger.debug("系统启动...")
    setup_dirs()

    if args.build_kg:
        run_kg_construction_pipeline(num_workers=args.workers)

    elif args.crawl_init or args.crawl_status or args.crawl:
        coordinator = Coordinator()
        if args.crawl_init:
            coordinator.initialize_queue(clear_existing=args.reset_queue)
        elif args.crawl_status:
            status = coordinator.get_queue_status()
        elif args.crawl:
            worker = Worker(use_screenshot=args.screenshot)
            asyncio.run(worker.run())

    elif args.process_text or args.process_image or args.process_all:
        pipeline = IndexingPipeline()
        if args.process_text or args.process_all:
            pipeline.process_directory(config.TEXT_DIR, 'text')
        if args.process_image or args.process_all:
            for image_dir in config.IMAGE_SOURCE_DIRS:
                pipeline.process_directory(image_dir, 'image')

    elif args.search_text or args.search_image:
        engine = SearchEngine()
        query_term = args.search_text or args.search_image
        if args.search_text:
            results = engine.search_by_text(query_term, args.limit)
        else:
            results = engine.search_by_image(query_term, args.limit)
        presenter = Presenter()
        presenter.display_search_results(results, query_term)

    elif args.ask or args.query_image:
        rag_engine = RAGEngine()
        presenter = Presenter()
        with presenter.console.status("[bold cyan]AI正在思考中...[/bold cyan]", spinner="dots"):
            report = rag_engine.ask(args.ask, image_path=args.query_image)
        presenter.display_rag_report(report)
    
    elapsed = time.time() - start_time
    logger.info(f"操作完成，总耗时: {elapsed:.2f} 秒")

if __name__ == "__main__":
    main()

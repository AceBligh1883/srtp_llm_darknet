# src/pipeline/processing.py
"""
核心处理流水线，负责编排处理流程
"""
import os
from tqdm import tqdm
import multiprocessing
from src.common import config
from src.common.logger import logger
from src.storage.database import DatabaseManager
from src.storage.index import IndexManager
from src.features.embedding import EmbeddingGenerator
from src.features.text_processing import TextProcessor
from src.features.image_processing import ImageProcessor

_pipeline_instance = None

def _initialize_worker():
    """
    这是每个工作进程启动时会且仅会调用一次的初始化函数。
    """
    global _pipeline_instance
    process_name = multiprocessing.current_process().name
    logger.info(f"正在初始化工作进程: {process_name}...")
    # 每个进程创建自己的、可复用的 ProcessingPipeline 实例
    _pipeline_instance = ProcessingPipeline()
    logger.info(f"工作进程 {process_name} 初始化成功。")

def _process_file_task(args):
    """
    这是每个工作进程执行具体任务的函数。
    它会复用由 _initialize_worker 创建的全局 pipeline 实例。
    """
    file_path, content_type = args
    try:
        # 直接使用已经初始化好的实例
        return _pipeline_instance.run_for_file(file_path, content_type)
    except Exception as e:
        logger.error(f"工作进程处理 '{file_path}' 时发生错误: {e}")
        return False
    
class ProcessingPipeline:
    """编排一个文件的完整处理与索引流程"""

    def __init__(self):
        self.db_manager = DatabaseManager(config.DB_PATH)
        self.index_manager = IndexManager(self.db_manager)
        self.embed_generator = EmbeddingGenerator()
        self.text_processor = TextProcessor()
        self.image_processor = ImageProcessor()
        logger.info("处理流水线初始化完成。")

    def run_for_file(self, file_path: str, content_type: str) -> bool:
        """对单个文件执行完整的处理和索引流程"""
        try:
            processed_text = None
            if content_type == 'text':
                if os.path.getsize(file_path) < config.MIN_TEXT_LENGTH:
                     logger.debug(f"文件 {os.path.basename(file_path)} 因大小过小被跳过。")
                     return False 

                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    raw_text = f.read()
                
                processed_text = self.text_processor.process(raw_text)
                if len(processed_text) < config.MIN_TEXT_LENGTH:
                    logger.info(f"文件 {os.path.basename(file_path)} 因有效内容过短(<{config.MIN_TEXT_LENGTH}字符)被跳过。")
                    return False 
                
            metadata = self.db_manager.create_metadata_from_file(file_path, content_type)
            doc_id = self.db_manager.save_metadata(metadata)
            if not doc_id:
                return False

            vector = None
            if content_type == 'text':
                vector = self.embed_generator.get_text_embedding(processed_text)
            elif content_type == 'image':
                pil_image = self.image_processor.process(file_path)
                if pil_image:
                    vector = self.embed_generator.get_image_embedding(pil_image)

            if vector is None:
                logger.error(f"文件向量生成失败: {file_path}")
                return False

            success = self.index_manager.index_document(doc_id, vector)
            return success
        except Exception as e:
            logger.error(f"处理文件 '{file_path}' 失败: {e}")
            return False

    def process_directory(self, directory: str, content_type: str):
        """处理指定目录下的所有文件"""
        if not os.path.exists(directory):
            logger.error(f"目录不存在: {directory}")
            return

        file_paths = []
        for root, _, files in os.walk(directory):
            for filename in files:
                if content_type == 'image' and not filename.lower().endswith(config.SUPPORTED_IMAGE_FORMATS):
                    continue
                if content_type == 'text' and not filename.lower().endswith('.txt'):
                    continue
                file_paths.append(os.path.join(root, filename))
        
        if not file_paths:
            logger.warning(f"在目录 {directory} 中未找到 {content_type} 类型的文件。")
            return

        num_files = len(file_paths)
        logger.info(f"在 {os.path.basename(directory)} 目录发现 {num_files} 个文件，使用 {config.PROCESSING_WORKERS} 个进程开始真正并发处理...")
        
        tasks = [(path, content_type) for path in file_paths]
        success_count = 0

        with multiprocessing.Pool(processes=config.PROCESSING_WORKERS, initializer=_initialize_worker) as pool:
            results_iterator = pool.imap_unordered(_process_file_task, tasks)
            
            pbar = tqdm(results_iterator, total=num_files, desc=f"高性能并发处理 {os.path.basename(directory)}")
            for success in pbar:
                if success:
                    success_count += 1
                    
        logger.info(f"目录 {os.path.basename(directory)} 处理完成。成功: {success_count}/{len(file_paths)}")

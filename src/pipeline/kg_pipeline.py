# src/pipeline/kg_pipeline.py
import os
import traceback
from src.common import config
from src.common.logger import logger
from src.analysis.graph_engine import KnowledgeGraphEngine
from src.storage.database import DatabaseManager
from src.services.knowledge_extractor import KnowledgeExtractor

def run_kg_construction_pipeline():
    """
    执行完整的知识图谱构建流程。
    Args:
        limit (Optional[int]): 如果提供，则只处理指定数量的文件，用于调试。
                               默认为None，处理所有文件。
    """
    logger.info("启动知识图谱构建流程...")
    extractor = KnowledgeExtractor()
    graph_engine = KnowledgeGraphEngine()
    db_manager = DatabaseManager(config.DB_PATH)

    try:
        text_files = db_manager.get_unprocessed_kg_files()
        if not text_files:
            logger.info("没有需要处理的新文件。知识图谱构建流程结束。")
            return
        logger.info(f"发现 {len(text_files)} 个新的文本文档需要处理。")

        for i, file_path in enumerate(text_files):
            doc_id = os.path.basename(file_path)
            logger.info(f"正在处理文档 {i+1}/{len(text_files)}: {doc_id}")
            
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                if len(content.strip()) < config.MIN_TEXT_LENGTH:
                    logger.warning(f"文档 {doc_id} 内容过短，跳过。")
                    db_manager.mark_kg_processed(file_path)
                    continue
                triples = extractor.extract(content, doc_id)
                if triples:
                    graph_engine.add_triples(triples, doc_id)
                db_manager.mark_kg_processed(file_path)

            except Exception as e:
                logger.error(f"处理文档 {doc_id} 时发生严重错误，已跳过该文件:\n{e}")
    
    except Exception as e:
        logger.critical(f"知识图谱构建流程发生致命错误: {e}\n{traceback.format_exc()}")
    finally:
        graph_engine.close()
        logger.info("知识图谱构建流程完成。")


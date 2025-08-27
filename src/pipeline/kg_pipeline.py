import os
import glob
import traceback
from typing import Optional

from src.common import config
from src.common.logger import logger
from src.analysis.graph_engine import KnowledgeGraphEngine

def run_kg_construction_pipeline():
    """
    执行完整的知识图谱构建流程。

    该流程会扫描所有文本文件，从中提取知识三元组，并将其存入Neo4j图数据库。

    Args:
        limit (Optional[int]): 如果提供，则只处理指定数量的文件，用于调试。
                               默认为None，处理所有文件。
    """
    logger.info("启动知识图谱构建流程...")
    engine = KnowledgeGraphEngine()
    
    try:
        text_files = glob.glob(os.path.join(config.TEXT_DIR, '**', '*.txt'), recursive=True)
        total_files = len(text_files)
        logger.info(f"发现 {total_files} 个文本文档可供处理。")

        for i, file_path in enumerate(text_files):
            doc_id = os.path.basename(file_path)
            logger.info(f"正在处理文档 {i+1}/{len(text_files)}: {doc_id}")
            
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                if len(content.strip()) < config.MIN_TEXT_LENGTH:
                    logger.warning(f"文档 {doc_id} 内容过短，跳过。")
                    continue
                triples = engine.extract_triples_from_text(content, doc_id)
                if triples:
                    engine.add_triples_to_graph(triples, doc_id)

            except Exception:
                error_details = traceback.format_exc()
                logger.error(f"处理文档 {doc_id} 时发生严重错误，已跳过该文件:\n{error_details}")
    
    except Exception as e:
        logger.critical(f"知识图谱构建流程发生致命错误，流程终止: {e}")
        error_details = traceback.format_exc()
        logger.critical(f"详细错误信息:\n{error_details}")

    finally:
        engine.close()
        logger.info("知识图谱构建流程完成。")


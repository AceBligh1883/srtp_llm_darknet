# src/pipeline/kg_pipeline.py
import os
import traceback
from tqdm import tqdm
from src.common import config
from src.common.logger import logger
from src.analysis.graph_engine import KnowledgeGraphEngine
from src.storage.database import DatabaseManager
from src.services.knowledge_extractor import KnowledgeExtractor

def run_kg_construction_pipeline():
    """
    执行完整的知识图谱构建流程。
    """
    logger.info("启动知识图谱构建流程...")
    extractor = KnowledgeExtractor()
    graph_engine = KnowledgeGraphEngine()
    db_manager = DatabaseManager(config.DB_PATH)
    kg_batch_size = config.KG_BATCH_SIZE

    try:
        files_to_process = db_manager.get_unprocessed_kg_files()
        if not files_to_process:
            logger.info("没有需要处理的新文件。知识图谱构建流程结束。")
            return
        total_files = len(files_to_process)
        logger.info(f"发现 {total_files} 个新文档。")
    
        with tqdm(total=total_files, desc="知识图谱构建") as progress_bar:
            for i in range(0, total_files, kg_batch_size):
                batch_paths = files_to_process[i:i + kg_batch_size]
                
                documents_for_batch = []
                processed_in_batch = []

                for file_path in batch_paths:
                    doc_id = os.path.basename(file_path)
                    try:
                        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                            content = f.read()
                        if len(content.strip()) < config.MIN_TEXT_LENGTH:
                            db_manager.mark_kg_processed(file_path)
                            processed_in_batch.append(file_path)
                        else:
                            documents_for_batch.append({'doc_id': doc_id, 'content': content})
                    except Exception as e:
                        logger.error(f"读取文件 {doc_id} 失败，跳过: {e}")
                        processed_in_batch.append(file_path)
                
                if not documents_for_batch:
                    progress_bar.update(len(batch_paths))
                    continue

                try:
                    extraction_results = extractor.extract_from_documents(documents_for_batch)
                    
                    for doc_info in documents_for_batch:
                        doc_id = doc_info['doc_id']
                        original_path = next(p for p in batch_paths if os.path.basename(p) == doc_id)
                        
                        triples = extraction_results.get(doc_id)
                        if triples:
                            graph_engine.add_triples(triples, doc_id)
                        
                        db_manager.mark_kg_processed(original_path)
                        processed_in_batch.append(original_path)

                except Exception as e:
                    logger.critical(f"处理批次时发生严重错误，跳过整个批次: {e}")
                
                progress_bar.update(len(batch_paths))

    except Exception as e:
        logger.critical(f"知识图谱构建流程发生致命错误: {e}\n{traceback.format_exc()}")
    finally:
        graph_engine.close()
        logger.info("知识图谱构建流程完成。")



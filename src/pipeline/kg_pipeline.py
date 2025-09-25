# src/pipeline/kg_pipeline.py (重构为调用Presenter版)
import os
import traceback
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.common import config
from src.common.logger import logger
from src.analysis.graph_engine import KnowledgeGraphEngine
from src.storage.database import DatabaseManager
from src.services.knowledge_extractor import KnowledgeExtractor
from src.ui.kg_presenter import KGPresenter

def process_batch_concurrently(batch_documents: list):
    """
    在单独的线程中处理一个批次的文档。
    """
    try:
        extractor = KnowledgeExtractor()
        results = extractor.extract_from_documents(batch_documents)
        return results, None
    except Exception as e:
        doc_ids = [doc['doc_id'] for doc in batch_documents]
        logger.error(f"线程中处理批次 {doc_ids} 时发生严重错误: {e}")
        return None, e

def run_kg_construction_pipeline(num_workers: int = 1):
    """
    执行完整的知识图谱构建流程，并调用Presenter进行可视化。
    """
    mode = "并行" if num_workers > 1 else "串行"
    logger.info(f"启动知识图谱构建流程 ({mode}模式, {num_workers}个worker)...")
    
    graph_engine = KnowledgeGraphEngine()
    db_manager = DatabaseManager(config.DB_PATH)
    presenter = KGPresenter() 
    
    BATCH_SIZE = getattr(config, 'KG_BATCH_SIZE', 10)

    try:
        files_to_process = db_manager.get_unprocessed_kg_files()
        if not files_to_process:
            logger.info("没有需要处理的新文件。流程结束。")
            return
        
        total_files = len(files_to_process)
        logger.info(f"发现 {total_files} 个新文档，将以 {BATCH_SIZE} 个文件为一个批次进行处理。")

        all_batches = []
        for i in range(0, total_files, BATCH_SIZE):
            batch_paths = files_to_process[i:i + BATCH_SIZE]
            documents_for_batch = []
            for file_path in batch_paths:
                doc_id = os.path.basename(file_path)
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                    if len(content.strip()) >= config.MIN_TEXT_LENGTH:
                        documents_for_batch.append({'doc_id': doc_id, 'content': content, 'path': file_path})
                    else:
                        db_manager.mark_kg_processed(file_path) 
                except Exception as e:
                    logger.error(f"读取文件 {doc_id} 失败，跳过: {e}")
            if documents_for_batch:
                all_batches.append(documents_for_batch)

        def handle_extraction_results(docs, results):
            for doc_info in docs:
                doc_id = doc_info['doc_id']
                triples = results.get(doc_id)
                
                if triples:
                    presenter.display_new_triples(doc_id, triples)
                    graph_engine.add_triples(triples, doc_id)
                
                db_manager.mark_kg_processed(doc_info['path'])

        with tqdm(total=total_files, desc="知识图谱构建") as progress_bar:
            if num_workers > 1:
                # 并行模式
                with ThreadPoolExecutor(max_workers=num_workers) as executor:
                    future_to_batch = {executor.submit(process_batch_concurrently, batch): batch for batch in all_batches}
                    for future in as_completed(future_to_batch):
                        batch_docs = future_to_batch[future]
                        try:
                            extraction_results, error = future.result()
                            if not error and extraction_results:
                                handle_extraction_results(batch_docs, extraction_results)
                        except Exception as exc:
                            logger.error(f"处理一个批次的结果时发生异常: {exc}")
                        progress_bar.update(len(batch_docs))
            else:
                # 串行模式
                extractor = KnowledgeExtractor()
                for batch_docs in all_batches:
                    try:
                        extraction_results = extractor.extract_from_documents(batch_docs)
                        handle_extraction_results(batch_docs, extraction_results)
                    except Exception as e:
                        logger.critical(f"处理批次时发生严重错误，跳过批次: {e}")
                    progress_bar.update(len(batch_docs))

    except Exception as e:
        logger.critical(f"知识图谱构建流程发生致命错误: {e}\n{traceback.format_exc()}")
    finally:
        graph_engine.close()
        logger.info("知识图谱构建流程完成。")


# processors/text_processor.py
import os
from src.logger import logger
from config import config
from src.analyzer.content_analyzer import DarknetAnalyzer
from src.utils.io_utils import load_texts, process_file

class TextProcessor:
    def __init__(self, analyzer: DarknetAnalyzer):
        self.analyzer = analyzer
        self.text_dir = config.TEXT_DIR

    def process(self) -> None:
        if not os.path.exists(self.text_dir):
            logger.error(f"文本目录不存在：{self.text_dir}")
            return
        texts = load_texts()
        if not texts:
            logger.error("未找到有效的文本文件")
            return
        
        logger.info(f"开始处理文本文件，共 {len(texts)} 个文件")
        processed_count = 0
        
        for index, file_item in enumerate(texts, 1):
            logger.info(f"处理文本进度: {index}/{len(texts)}")
            success = process_file(file_item, self.analyzer, process_image=False)
            if success:
                processed_count += 1
        
        logger.info(f"文本处理完成，成功处理 {processed_count}/{len(texts)} 个文件")

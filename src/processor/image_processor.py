# processors/image_processor.py
import os
from src.logger import logger
from config import config
from src.analyzer.content_analyzer import DarknetAnalyzer
from src.utils.io_utils import process_file

class ImageProcessor:
    def __init__(self, analyzer: DarknetAnalyzer):
        self.analyzer = analyzer

    def process(self) -> None:
        if not os.path.exists(config.IMAGE_DIR):
            logger.error(f"图像目录不存在：{config.IMAGE_DIR}")
            return
            
        image_files = [f for f in os.listdir(config.IMAGE_DIR) 
                      if any(f.lower().endswith(ext) for ext in config.SUPPORTED_IMAGE_FORMATS)]
        
        if not image_files:
            logger.error("未找到有效的图像文件")
            return
        
        logger.info(f"开始处理图像文件，共 {len(image_files)} 个文件")
        processed_count = 0
        
        for index, image_file in enumerate(image_files, 1):
            logger.info(f"处理图像进度: {index}/{len(image_files)}")
            file_item = {"filename": image_file, "content": ""}
            success = process_file(file_item, self.analyzer, process_image=True)
            if success:
                processed_count += 1
        
        logger.info(f"图像处理完成，成功处理 {processed_count}/{len(image_files)} 个文件")

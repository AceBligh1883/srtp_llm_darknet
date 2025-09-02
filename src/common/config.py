# src/common/config.py
"""
项目统一配置文件。
"""
import os
from typing import List, Tuple

# 导入私有配置
try:
    from .private_config import *
except ImportError:
    raise ImportError(
        "错误：无法找到 'private_config.py' 文件。"
    )

# --- 公共配置 ---

# 基础路径配置 
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, 'data')
DB_PATH = os.path.join(DATA_DIR, 'darknet.db')

# 输入数据源
TEXT_DIR = os.path.join(DATA_DIR, 'input', 'text')
IMAGE_DIR = os.path.join(DATA_DIR, 'input', 'images') 
SCREENSHOTS_DIR = os.path.join(DATA_DIR, 'input', 'screenshots')
VIDEOS_DIR = os.path.join(DATA_DIR, 'input', 'videos')
FILE_DIR = os.path.join(DATA_DIR, 'input', 'files')

# 图像处理任务的源目录列表
IMAGE_SOURCE_DIRS: List[str] = [
    IMAGE_DIR,
    SCREENSHOTS_DIR
]
# 支持的图像文件扩展名
SUPPORTED_IMAGE_FORMATS: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp")

# 模型配置
MODEL_NAME: str = "google/siglip-base-patch16-256-multilingual"
VECTOR_DIM: int = 768 

RERANKER_MODEL: str = "BAAI/bge-reranker-v2-m3"

# LLM (用于RAG)
RAG_TOP_K: int = 10
DEFAULT_MODEL: str = "gemini-2.5-pro"

# 性能与并发配置
MAX_TOKENS_PER_BATCH: int = 65536
MIN_TEXT_LENGTH: int = 100
#MAX_TEXT_CHARS: int = 8192
MAX_IMAGE_PIXELS: int = 2048 * 2048
MAX_ITEMS_PER_BATCH = 32 

# 存储与数据库配置
ES_INDEX: str = "darknet_multimodal_index"
ES_TIMEOUT: int = 30
REDIS_DB: int = 0
TASK_QUEUE: str = "darknet_tasks"
VISITED_SET: str = "visited_urls"

# 爬虫模块配置
SEED_URLS: List[str] = [
    "http://6nhmgdpnyoljh5uzr5kwlatx2u3diou4ldeommfxjz3wkhalzgjqxzqd.onion/",
]
USER_AGENT: str = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Firefox/91.0")
MAX_CONCURRENCY: int = 8
SCREENSHOT_WORKERS: int = 2
MAX_DEPTH: int = 5
REQUEST_TIMEOUT: int = 60
RATE_LIMIT: int = 10

# 日志配置
LOG_LEVEL: str = "INFO"
LOG_FORMAT: str = "%(asctime)s [%(levelname)-8s] %(message)s"
DATE_FORMAT: str = "%Y-%m-%d %H:%M:%S"

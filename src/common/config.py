# src/common/config.py
"""
统一的、适配原有设置的配置文件
"""
import os
from typing import List, Tuple

# --- 1. 基础目录配置 ---
# 项目根目录
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# 主数据目录
DATA_DIR = os.path.join(BASE_DIR, 'data')

# 输入数据目录 (保留您的详细分类)
TEXT_DIR = os.path.join(DATA_DIR, 'input', 'text')
IMAGE_DIR = os.path.join(DATA_DIR, 'input', 'images') # 保持 'images' 与您原有的一致
SCREENSHOTS_DIR = os.path.join(DATA_DIR, 'input', 'screenshots')
VIDEOS_DIR = os.path.join(DATA_DIR, 'input', 'videos')
FILE_DIR = os.path.join(DATA_DIR, 'input', 'files')

# 数据库路径 (保留您的设置)
DB_PATH = os.path.join(DATA_DIR, 'darknet.db')


# --- 2. 模型与处理配置 ---
# 使用的模型是 openai/clip-vit-base-patch16，其向量维度是 512
MODEL_NAME = "OFA-Sys/chinese-clip-vit-base-patch16"
VECTOR_DIM = 512 # 这个值由模型决定，不应随意更改

# 文本处理时的最小长度（字节）
MIN_TEXT_LENGTH = 100


# --- 3. Elasticsearch 配置 (新增) ---
ES_HOST = "127.0.0.1"
ES_PORT = 9200
ES_TIMEOUT = 30
ES_INDEX = "darknet_multimodal_index"


# --- 4. 日志配置 (保留您的设置) ---
LOG_LEVEL = "INFO"  # DEBUG, INFO, WARNING, ERROR, CRITICAL
LOG_FORMAT = "%(asctime)s [%(levelname)-8s] %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


# --- 5. 文件格式配置 (保留您的设置) ---
SUPPORTED_IMAGE_FORMATS: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp")


# --- 6. 爬虫配置 (完整保留您的设置) ---
# Redis 配置
REDIS_HOST: str = "localhost"
REDIS_PORT: int = 6379
REDIS_DB: int = 0
TASK_QUEUE: str = "darknet_tasks"
VISITED_SET: str = "visited_urls"

# 网络与代理配置
TOR_PROXY: str = "socks5://127.0.0.1:9150"
USER_AGENT: str = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Firefox/91.0")

# 并发与性能参数
MAX_CONCURRENCY: int = 8          # 最大并发请求数
SCREENSHOT_WORKERS: int = 2       # 浏览器池中截图实例数
MAX_DEPTH: int = 5                # 最大爬取深度
REQUEST_TIMEOUT: int = 60         # 请求超时（秒）
RATE_LIMIT: int = 10              # 请求速率限制 (sleep = 1 / RATE_LIMIT 秒)

# Selenium 配置
GECKODRIVER_PATH: str = "D:\\Tor Browser\\Browser\\geckodriver.exe"
TOR_BROWSER_BINARY: str = "D:\\Tor Browser\\Browser\\firefox.exe"

# 种子URL
SEED_URLS: List[str] = [
    "http://6nhmgdpnyoljh5uzr5kwlatx2u3diou4ldeommfxjz3wkhalzgjqxzqd.onion/",
]

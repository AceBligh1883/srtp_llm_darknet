# config/__init__.py
from dataclasses import dataclass, field
from typing import List, Tuple

@dataclass
class config:
    """配置类"""
    # 基础目录配置
    TEXT_DIR: str = "data/input/text"
    IMAGE_DIR: str = "data/input/images"
    SCREENSHOTS_DIR: str = "data/input/screenshots"
    VIDEOS_DIR: str = "data/input/videos"
    FILE_DIR: str = "data/input/files"
    OUTPUT_DIR: str = "data/output/json" 
    
    # 模型配置
    DEFAULT_MODEL: str = "qwen2.5:14b"
    ALIGNED_DIM: int = 512
    
    # 文本处理配置
    MAX_TEXT_LENGTH: int = 1024
    MIN_TEXT_LENGTH: int = 100
    
    # 日志配置
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "%(asctime)s [%(levelname)-8s] %(message)s"
    DATE_FORMAT: str = "%Y-%m-%d %H:%M:%S"
    
    # 文件格式配置
    SUPPORTED_IMAGE_FORMATS: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp")
    
    # 爬虫配置
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
    
    # 数据库配置
    DB_PATH: str = "data/darknet.db"

    # 种子URL
    SEED_URLS: List[str] = field(default_factory=lambda: [
        "http://6nhmgdpnyoljh5uzr5kwlatx2u3diou4ldeommfxjz3wkhalzgjqxzqd.onion/",
    ])

# 创建默认配置实例
config = config()


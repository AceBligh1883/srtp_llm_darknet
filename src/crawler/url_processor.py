# src/crawler/url_processor.py
"""
URL处理工具模块
"""
from urllib.parse import urlparse, urlunparse, urljoin

class URLProcessor:
    """URL处理工具类"""
    @staticmethod
    def normalize(url: str) -> str:
        """标准化 URL，自动添加协议并转换为小写"""
        url = url.strip()
        if not url.lower().startswith("http"):
            url = "http://" + url
        return url.lower()

    @staticmethod
    def is_onion(url: str) -> bool:
        """判断是否为 .onion 域名"""
        try:
            return urlparse(url).netloc.endswith(".onion")
        except Exception:
            return False

    @staticmethod
    def canonicalize(url: str) -> str:
        """规范化 URL，用于去重（忽略末尾斜杠、统一大小写）"""
        try:
            parsed = urlparse(url)
            scheme = parsed.scheme.lower()
            netloc = parsed.netloc.lower()
            path = parsed.path.rstrip('/') or '/'
            return urlunparse((scheme, netloc, path, parsed.params, parsed.query, ""))
        except Exception:
            return ""

    @staticmethod
    def join(base_url: str, relative_url: str) -> str:
        """合并基础URL和相对URL"""
        try:
            return urljoin(base_url, relative_url)
        except Exception:
            return ""

    @staticmethod
    def get_domain(url: str) -> str:
        """从URL中提取域名"""
        try:
            return urlparse(url).netloc
        except Exception:
            return ""

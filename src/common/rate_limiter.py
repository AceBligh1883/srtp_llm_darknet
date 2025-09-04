# src/common/utils.py
import time
import functools
from src.common.logger import logger

def rate_limited(seconds: float):
    """
    一个装饰器工厂，用于为函数添加速率限制。
    它确保两次函数调用之间至少间隔指定的秒数。

    Args:
        seconds (float): 最小的调用间隔时间（秒）。
    """
    last_call_info = {'time': 0}

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            elapsed = time.time() - last_call_info['time']
            
            if elapsed < seconds:
                wait_time = seconds - elapsed
                logger.info(f"速率限制：等待 {wait_time:.2f} 秒后继续...")
                time.sleep(wait_time)
            
            result = func(*args, **kwargs)
            last_call_info['time'] = time.time()
            return result
        return wrapper
    return decorator

def format_exception_detail(e: Exception) -> str:
    """
    将异常对象格式化为包含类型、参数和根本原因的详细字符串。
    """
    # 基础信息
    error_type = type(e).__name__
    error_args = e.args
    
    # 尝试获取 requests 库的底层 urllib3 错误
    cause = getattr(e, '__cause__', None)
    cause_detail = ""
    if cause:
        cause_detail = f"\n  - Underlying Cause: {type(cause).__name__}{cause.args}"

    # 尝试获取请求对象信息
    request_info = ""
    request = getattr(e, 'request', None)
    if request:
        request_info = f"\n  - Request Method: {request.method}\n  - Request URL: {request.url}"

    return (
        f"{error_type} - Args: {error_args}{request_info}{cause_detail}"
    )
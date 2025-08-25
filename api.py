# api.py

import os
import shutil
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from typing import List, Optional

from src.common.logger import logger
from src.search.engine import SearchEngine
from src.search.rag_engine import RAGEngine
from src.common.data_models import SearchResult 

# --- 初始化应用和引擎 ---

engines = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    定义应用的生命周期事件。
    在 'yield' 之前的代码会在应用启动时执行。
    在 'yield' 之后的代码会在应用关闭时执行。
    """
    logger.info("应用启动... 正在加载核心引擎...")
    try:
        engines["search_engine"] = SearchEngine()
        engines["rag_engine"] = RAGEngine()
        logger.info("引擎加载成功。API服务已准备就绪。")
    except Exception as e:
        logger.error(f"引擎加载失败: {e}", exc_info=True)
    
    yield

    logger.info("应用关闭...")
    engines.clear()
    logger.info("资源已清理。")

servers = [
    {
        "url": "http://10.208.95.76:8000",
        "description": "本地开发服务器"
    }
]

app = FastAPI(
    title="暗网多模态检索引擎 API", 
    description="一个提供文本搜索、图像搜索和RAG问答功能的API服务。",
    version="1.0.0",
    lifespan=lifespan,
    servers=servers
)


@app.get("/", summary="健康检查", description="检查API服务是否正在运行。")
def health_check():
    """一个简单的端点，用于确认服务是否在线。"""
    return {"status": "ok", "message": "API is running"}

@app.post("/search-text", summary="文本搜索", response_model=List[SearchResult])
async def api_search_text(query: str = Form(...), top_k: int = Form(10)):
    """
    根据文本查询执行混合搜索。
    这是Dify可以调用的一个工具。
    """
    search_engine = engines.get("search_engine")
    if not search_engine:
        raise HTTPException(status_code=503, detail="服务暂时不可用，检索引擎未加载。")
    
    logger.info(f"API收到文本搜索请求: query='{query}', top_k={top_k}")
    try:
        results = search_engine.search_by_text(query, top_k)
        return results
    except Exception as e:
        logger.error(f"文本搜索API处理失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="内部服务器错误")

@app.post("/search-image", summary="图像搜索 (文搜图或图搜图)", response_model=List[SearchResult])
async def api_search_image(
    query_text: Optional[str] = Form(None), 
    image_file: Optional[UploadFile] = File(None),
    top_k: int = Form(10)
):
    """
    根据文本或上传的图像文件，在图像库中进行搜索。
    - 如果提供了`query_text`，执行文搜图。
    - 如果提供了`image_file`，执行图搜图。
    两者必须提供一个。
    """
    search_engine = engines.get("search_engine")
    if not search_engine:
        raise HTTPException(status_code=503, detail="服务暂时不可用，检索引擎未加载。")
    
    if not query_text and not image_file:
        raise HTTPException(status_code=400, detail="必须提供 'query_text' 或 'image_file'。")

    query_input = ""
    temp_file_path = "" # 确保变量已定义
    try:
        if image_file:
            temp_dir = "temp_uploads"
            os.makedirs(temp_dir, exist_ok=True)
            temp_file_path = os.path.join(temp_dir, image_file.filename)
            
            with open(temp_file_path, "wb") as buffer:
                shutil.copyfileobj(image_file.file, buffer)
            
            query_input = temp_file_path
            logger.info(f"API收到图搜图请求: file='{image_file.filename}', top_k={top_k}")
            
        elif query_text:
            query_input = query_text
            logger.info(f"API收到文搜图请求: query='{query_text}', top_k={top_k}")

        results = search_engine.search_by_image(query_input, top_k)
        
        return results

    except Exception as e:
        logger.error(f"图像搜索API处理失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="内部服务器错误")
    finally:
        if image_file and temp_file_path and os.path.exists(temp_file_path):
            os.remove(temp_file_path)


@app.post("/ask-rag", summary="RAG问答", response_model=str)
async def api_ask_rag(question: str = Form(...)):
    """
    向RAG引擎提问，获取基于本地知识库的智能回答。
    """
    rag_engine = engines.get("rag_engine")
    if not rag_engine:
        raise HTTPException(status_code=503, detail="服务暂时不可用，RAG引擎未加载。")
        
    logger.info(f"API收到RAG问答请求: question='{question}'")
    try:
        answer = rag_engine.ask(question)
        return answer
    except Exception as e:
        logger.error(f"RAG问答API处理失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="内部服务器错误")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)

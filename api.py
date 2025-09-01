# api.py
import base64
import os
import shutil
import json
from contextlib import asynccontextmanager
from typing import List, Optional, Dict, Any
from dataclasses import asdict
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from src.search.engine import SearchEngine
from src.search.rag_engine import RAGEngine
from src.common.logger import logger
from src.common.data_models import SearchResult 

def _image_to_base64(file_path: str) -> Optional[str]:
    """将图片文件转换为Base64编码的Data URI。"""
    if not os.path.exists(file_path):
        return None
    try:
        # 推断MIME类型
        ext = os.path.splitext(file_path)[1].lower()
        mime_types = {'.jpg': 'image/jpeg', '.jpeg': 'image/jpeg', '.png': 'image/png', '.gif': 'image/gif'}
        mime_type = mime_types.get(ext, 'application/octet-stream')

        with open(file_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
        return f"data:{mime_type};base64,{encoded_string}"
    except Exception as e:
        logger.error(f"转换图片到Base64失败: {file_path}, 错误: {e}")
        return None
    
def _read_snippet_from_file(file_path: str, max_len: int = 300) -> str:
    """从文件路径安全地读取一个内容片段。"""
    if not os.path.exists(file_path):
        return "[内容文件不存在]"
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read(max_len + 1)
        if len(content) > max_len:
            return content[:max_len] + "..."
        return content
    except Exception as e:
        return f"[读取文件时出错: {e}]"

engines: Dict[str, Any] = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI应用的生命周期事件。"""
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
    title="多模态情报分析API",
    description="为Dify工作流提供文本搜索、图像搜索和RAG问答的工具接口。",
    version="1.0.0",
    lifespan=lifespan,
    servers=servers
)

@app.post("/search-by-text", summary="文本多模态搜索")
async def api_search_by_text(query: str = Form(...), top_k: int = Form(10)):
    """
    接受文本输入。
    """
    search_engine: SearchEngine = engines.get("search_engine")
    if not search_engine:
        raise HTTPException(status_code=503, detail="服务暂时不可用，检索引擎未加载。")
    
    logger.info(f"API收到文本搜索请求: query='{query}', top_k={top_k}")
    try:
        results = search_engine.search_by_text(query, top_k)
        
        enriched_results: List[Dict[str, Any]] = []
        for res in results:
            res_dict = asdict(res)
            if res.metadata.content_type == 'text':
                res_dict['content_snippet'] = _read_snippet_from_file(res.metadata.file_path)
            else:
                res_dict['base64_data'] = _image_to_base64(res.metadata.file_path)
            enriched_results.append(res_dict)
        enriched_results = jsonable_encoder(enriched_results)    
        return JSONResponse(content=enriched_results)
    except Exception as e:
        logger.error(f"文本搜索API处理失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="内部服务器错误")
    
@app.post("/search-by-image", summary="图像多模态搜索")
async def api_search_by_image(image_file: UploadFile = File(...), top_k: int = Form(10)):
    """
    接受图像文件输入。
    """
    search_engine: SearchEngine = engines.get("search_engine")
    if not search_engine:
        raise HTTPException(status_code=503, detail="服务暂时不可用，检索引擎未加载。")

    temp_file_path = ""
    try:
        temp_dir = "temp_uploads"; os.makedirs(temp_dir, exist_ok=True)
        temp_file_path = os.path.join(temp_dir, image_file.filename)
        with open(temp_file_path, "wb") as buffer: shutil.copyfileobj(image_file.file, buffer)
        results = search_engine.search_by_image(temp_file_path, top_k)
        enriched_results: List[Dict[str, Any]] = []
        for res in results:
            res_dict = asdict(res)
            if res.metadata.content_type == 'image':
                res_dict['base64_data'] = _image_to_base64(res.metadata.file_path)
            enriched_results.append(res_dict)
        
        enriched_results = jsonable_encoder(enriched_results)
        return JSONResponse(content=enriched_results)

    except Exception as e:
        logger.error(f"图像搜索API处理失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="内部服务器错误")
    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            os.remove(temp_file_path)

@app.post("/ask-rag", summary="RAG问答")
async def api_ask_rag(question: str = Form(...)):
    """向RAG引擎提问，获取基于本地知识库的智能回答。"""
    rag_engine: RAGEngine = engines.get("rag_engine")
    if not rag_engine:
        raise HTTPException(status_code=503, detail="服务暂时不可用，RAG引擎未加载。")
        
    logger.info(f"API收到RAG问答请求: question='{question}'")
    try:
        answer = rag_engine.ask(question)
        return JSONResponse(content={"answer": answer})
    except Exception as e:
        logger.error(f"RAG问答API处理失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="内部服务器错误")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

# api_server.py
import os
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional, List
import uvicorn
import urllib.parse

from src.common.logger import logger
from src.common.data_models import RAGReport, SearchResult
from src.search.rag_engine import RAGEngine

# --- 初始化 ---
app = FastAPI(
    title="多模态分析与检索系统 API",
    description="提供RAG问答、内容检索等功能的后端服务。",
    version="1.0.0",
)
rag_engine = RAGEngine()

ALLOWED_PATH_PREFIX = os.path.abspath("data/input")


class RAGResponse(BaseModel):
    report: RAGReport

# --- API 端点定义 ---
@app.post("/api/rag_query", response_model=RAGResponse, summary="执行RAG问答")
async def handle_rag_query(
    question: Optional[str] = Form(None), 
    query_image: Optional[UploadFile] = File(None)
):
    """
    接收文本和/或图片查询，执行完整的RAG流程。
    - **question**: 用户的文本问题。
    - **query_image**: 用户上传的查询图片。
    """
    if not question and not query_image:
        raise HTTPException(status_code=400, detail="必须提供问题或图片。")

    image_path = None
    try:
        if query_image:
            # 将上传的图片保存到临时位置
            temp_dir = "data/temp"
            os.makedirs(temp_dir, exist_ok=True)
            image_path = os.path.join(temp_dir, query_image.filename)
            with open(image_path, "wb") as buffer:
                buffer.write(await query_image.read())
            logger.info(f"接收到上传图片: {query_image.filename}")
        
        logger.info(f"开始处理RAG查询: question='{question}'")
        # 调用我们稳定可靠的后端引擎
        report_data = rag_engine.ask(question, image_path=image_path)
        return RAGResponse(report=report_data)

    except Exception as e:
        logger.critical(f"RAG API处理时发生严重错误: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"内部服务器错误: {e}")
    finally:
        # 清理临时文件
        if image_path and os.path.exists(image_path):
            os.remove(image_path)

@app.get("/api/media/{file_path:path}", summary="获取媒体文件")
async def get_media_file(file_path: str):
    """
    安全地提供服务器上的文件给前端。
    """
    try:
        # URL解码文件路径
        decoded_path = urllib.parse.unquote(file_path)
        
        # [安全检查] 确保请求的路径在我们允许的范围内
        full_path = os.path.abspath(decoded_path)
        if not full_path.startswith(ALLOWED_PATH_PREFIX):
            raise HTTPException(status_code=403, detail="禁止访问此路径。")

        if os.path.exists(full_path):
            return FileResponse(full_path)
        else:
            raise HTTPException(status_code=404, detail="文件未找到。")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"读取文件时出错: {e}")


# --- 运行服务器 ---
if __name__ == "__main__":
    logger.info("启动FastAPI服务器...")
    uvicorn.run(app, host="0.0.0.0", port=8000)


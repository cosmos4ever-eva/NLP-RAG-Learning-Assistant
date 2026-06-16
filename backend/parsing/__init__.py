"""
课程资料解析模块
负责: PDF/PPT 文本提取 → 清洗 → chunk 切分 → 来源标注
"""
import os
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, HTTPException

from shared.schemas import ParseResponse, ChunkItem
from .parser import parse_all_course_files, parse_single_file, list_course_files

router = APIRouter()

# 内存存储：已解析的 chunks
_all_chunks: dict[str, ChunkItem] = {}
_file_texts: dict[str, str] = {}


@router.post("/parse-all", response_model=list[ParseResponse])
async def parse_all_documents():
    """
    批量解析 course_materials/ 目录下所有 PDF
    返回每个文件的解析结果（全文本 + chunks）
    """
    try:
        results = parse_all_course_files()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"解析失败: {str(e)}")

    if not results:
        return []

    # 存入内存
    global _all_chunks, _file_texts
    responses = []
    for filename, data in results.items():
        _file_texts[filename] = data["full_text"]
        for chunk in data["chunks"]:
            _all_chunks[chunk.chunk_id] = chunk

        responses.append(ParseResponse(
            filename=filename,
            full_text=data["full_text"][:2000] + "..." if len(data["full_text"]) > 2000 else data["full_text"],
            chunks=data["chunks"],
            chunk_count=data["chunk_count"],
        ))

    return responses


@router.post("/upload", response_model=ParseResponse)
async def upload_and_parse(file: UploadFile = File(...)):
    """
    上传单个 PDF 并解析
    """
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="仅支持 PDF 文件")

    # 保存临时文件
    tmp_path = Path(f"/tmp/{file.filename}")
    try:
        content = await file.read()
        with open(tmp_path, "wb") as f:
            f.write(content)

        result = parse_single_file(tmp_path)

        # 存入内存
        global _all_chunks, _file_texts
        _file_texts[result["filename"]] = result["full_text"]
        for chunk in result["chunks"]:
            _all_chunks[chunk.chunk_id] = chunk

        return ParseResponse(
            filename=result["filename"],
            full_text=result["full_text"][:2000] + "..." if len(result["full_text"]) > 2000 else result["full_text"],
            chunks=result["chunks"],
            chunk_count=result["chunk_count"],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"解析失败: {str(e)}")
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


@router.get("/chunks", response_model=list[ChunkItem])
async def list_chunks(source: str = None):
    """
    列出所有已解析的 chunk，可按来源文件筛选
    """
    global _all_chunks
    chunks = list(_all_chunks.values())
    if source:
        chunks = [c for c in chunks if source in c.source]
    return sorted(chunks, key=lambda c: (c.source, c.page, c.chunk_index))


@router.get("/chunks/{chunk_id}", response_model=ChunkItem)
async def get_chunk(chunk_id: str):
    """根据 ID 获取单个 chunk"""
    global _all_chunks
    chunk = _all_chunks.get(chunk_id)
    if not chunk:
        raise HTTPException(status_code=404, detail=f"Chunk {chunk_id} 不存在")
    return chunk


@router.get("/files")
async def list_files():
    """列出 course_materials/ 下所有 PDF 文件"""
    files = list_course_files()
    return [{"name": f.name, "size": f.stat().st_size} for f in files]

"""
课程资料解析引擎
负责: PDF 文本提取 → 清洗 → 分块
"""
import re
import os
import json
import posixpath
import xml.etree.ElementTree as ET
from zipfile import ZipFile
from pathlib import Path
from pypdf import PdfReader

from shared.schemas import ChunkItem


# 项目根目录（backend/ 的上级）
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
COURSE_MATERIALS_DIR = PROJECT_ROOT / "course_materials"
MANUAL_SUMMARIES_PATH = COURSE_MATERIALS_DIR / "manual_page_summaries.json"

# chunk 目标大小（字符数）
CHUNK_MIN = 150
CHUNK_MAX = 500


def list_course_files() -> list[Path]:
    """列出 course_materials/ 下所有 PDF 文件"""
    if not COURSE_MATERIALS_DIR.exists():
        return []
    return sorted(p for p in COURSE_MATERIALS_DIR.iterdir()
                  if p.is_file() and p.suffix.lower() in {'.pdf', '.pptx'})


def extract_text_from_pptx(file_path: Path) -> tuple[str, list[dict]]:
    """Read slide text in presentation order, preserving slide numbers.

    Image-only slides require OCR and are intentionally not invented here.
    """
    ns = {'p': 'http://schemas.openxmlformats.org/presentationml/2006/main',
          'a': 'http://schemas.openxmlformats.org/drawingml/2006/main',
          'r': 'http://schemas.openxmlformats.org/officeDocument/2006/relationships'}
    pages = []
    with ZipFile(file_path) as archive:
        rels = ET.fromstring(archive.read('ppt/_rels/presentation.xml.rels'))
        targets = {r.attrib['Id']: r.attrib['Target'] for r in rels
                   if r.attrib.get('TargetMode') != 'External'}
        root = ET.fromstring(archive.read('ppt/presentation.xml'))
        for page, slide in enumerate(root.findall('p:sldIdLst/p:sldId', ns), 1):
            target = targets[slide.attrib['{' + ns['r'] + '}id']]
            part = target.lstrip('/') if target.startswith('/') else posixpath.normpath('ppt/' + target)
            xml = ET.fromstring(archive.read(part))
            paragraphs = [''.join(t.text or '' for t in p.findall('.//a:t', ns))
                          for p in xml.findall('.//a:p', ns)]
            text = clean_text('\n'.join(paragraphs))
            if text:
                pages.append({'page': page, 'text': text})
    return '\n\n'.join(p['text'] for p in pages), pages


def extract_document(file_path: Path):
    if file_path.suffix.lower() == '.pptx':
        _, pages = extract_text_from_pptx(file_path)
    elif file_path.suffix.lower() == '.pdf':
        _, pages = extract_text_from_pdf(file_path)
    else:
        raise ValueError('Unsupported course document format: ' + file_path.suffix)
    pages = _apply_manual_page_summaries(file_path, pages)
    return '\n\n'.join(page['text'] for page in pages), pages


def _apply_manual_page_summaries(file_path: Path, pages: list[dict]) -> list[dict]:
    """Append reviewed descriptions for diagrams that PDF text extraction cannot see."""
    if not MANUAL_SUMMARIES_PATH.exists():
        return pages
    with open(MANUAL_SUMMARIES_PATH, encoding="utf-8") as handle:
        summaries = json.load(handle).get(file_path.name, {})
    if not summaries:
        return pages

    by_page = {page["page"]: dict(page) for page in pages}
    for page_number, summary in summaries.items():
        page = int(page_number)
        existing = by_page.get(page, {"page": page, "text": ""})
        existing["text"] = clean_text(
            "\n\n".join(part for part in (existing["text"], summary) if part)
        )
        by_page[page] = existing
    return [by_page[page] for page in sorted(by_page)]


def extract_text_from_pdf(file_path: Path) -> tuple[str, list[dict]]:
    """
    从 PDF 逐页提取文本
    返回: (完整文本, [{page, text}, ...])
    """
    reader = PdfReader(str(file_path))
    pages_text = []
    full_text_parts = []

    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        # 基础清洗：保留有用字符
        text = clean_text(text)
        if text.strip():
            pages_text.append({"page": i + 1, "text": text})
            full_text_parts.append(text)

    full_text = "\n\n".join(full_text_parts)
    return full_text, pages_text


def clean_text(text: str) -> str:
    """文本清洗"""
    # 移除多余空白行（3 个以上换行压缩为 2 个）
    text = re.sub(r"\n{3,}", "\n\n", text)
    # 移除行内多余空格
    text = re.sub(r" {2,}", " ", text)
    # 移除常见页眉页脚数字（如页码单独一行）
    text = re.sub(r"^\d{1,3}\s*$", "", text, flags=re.MULTILINE)
    # 统一中文标点周围的空格
    text = re.sub(r"\s*([，。！？；：、])\s*", r"\1", text)
    return text.strip()


def _split_into_sentences(text: str) -> list[str]:
    """按句号、分号等切分长段落"""
    # 按中英文标点切分
    parts = re.split(r"(?<=[。！？；;!?\n])\s*", text)
    result = []
    for part in parts:
        part = part.strip()
        if part:
            result.append(part)
    return result


def chunk_text(text: str, source: str, page: int, start_index: int = 0) -> list[ChunkItem]:
    """
    将文本切分为 chunk
    策略: 先按段落切 → 长段落再按句切
    """
    chunks = []
    paragraphs = text.split("\n\n")
    chunk_idx = start_index

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        if len(para) <= CHUNK_MAX:
            # 段落长度合适，直接作为一个 chunk
            chunk_id = f"{source}_p{page}_c{chunk_idx}"
            chunks.append(ChunkItem(
                chunk_id=chunk_id,
                text=para,
                source=source,
                page=page,
                chunk_index=chunk_idx,
            ))
            chunk_idx += 1
        else:
            # 段落太长，按句切分
            sentences = _split_into_sentences(para)
            current_chunk = ""
            for sent in sentences:
                if len(current_chunk) + len(sent) < CHUNK_MAX:
                    current_chunk += sent
                else:
                    if current_chunk.strip() and len(current_chunk.strip()) >= CHUNK_MIN:
                        chunk_id = f"{source}_p{page}_c{chunk_idx}"
                        chunks.append(ChunkItem(
                            chunk_id=chunk_id,
                            text=current_chunk.strip(),
                            source=source,
                            page=page,
                            chunk_index=chunk_idx,
                        ))
                        chunk_idx += 1
                    current_chunk = sent
            # 处理最后一段
            if current_chunk.strip() and len(current_chunk.strip()) >= CHUNK_MIN:
                chunk_id = f"{source}_p{page}_c{chunk_idx}"
                chunks.append(ChunkItem(
                    chunk_id=chunk_id,
                    text=current_chunk.strip(),
                    source=source,
                    page=page,
                    chunk_index=chunk_idx,
                ))
                chunk_idx += 1

    return chunks


def parse_all_course_files() -> dict:
    """
    批量解析 course_materials/ 下所有 PDF
    返回: {filename: {full_text, chunks, chunk_count}}
    """
    results = {}
    files = list_course_files()

    for file_path in files:
        source_short = file_path.stem  # e.g., "CS-NLP-L2 Text Preprocession"
        # 简化文件名作为 source 标识
        source_short = source_short.replace(" ", "_")[:50]

        full_text, pages_text = extract_document(file_path)

        all_chunks = []
        for page_info in pages_text:
            page_chunks = chunk_text(
                page_info["text"],
                source=source_short,
                page=page_info["page"],
                start_index=len(all_chunks),
            )
            all_chunks.extend(page_chunks)

        results[file_path.name] = {
            "filename": file_path.name,
            "full_text": full_text,
            "chunks": all_chunks,
            "chunk_count": len(all_chunks),
        }

    return results


def parse_single_file(file_path: Path) -> dict:
    """解析单个 PDF 文件"""
    source_short = file_path.stem.replace(" ", "_")[:50]

    full_text, pages_text = extract_document(file_path)

    all_chunks = []
    for page_info in pages_text:
        page_chunks = chunk_text(
            page_info["text"],
            source=source_short,
            page=page_info["page"],
            start_index=len(all_chunks),
        )
        all_chunks.extend(page_chunks)

    return {
        "filename": file_path.name,
        "full_text": full_text,
        "chunks": all_chunks,
        "chunk_count": len(all_chunks),
    }

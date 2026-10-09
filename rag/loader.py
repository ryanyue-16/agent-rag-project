from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from pypdf import PdfReader

logger = logging.getLogger(__name__)

def load_pdf_documents(
    pdf_dir:str | Path,
) -> list[dict[str, Any]]:

    directory = Path(pdf_dir)
    pdf_paths = sorted(directory.glob("*.pdf"))

    logger.info(
        "扫描 PDF 目录：directory=%s file_count=%d", 
        directory,
        len(pdf_paths),
    )

    if not pdf_paths:
        raise ValueError(
            f"没有在 {directory} 中找到 PDF 文件。"
             "请把至少一个 .pdf 文件放入该目录。")

    documents:list[dict[str, Any]] = []

    for pdf_path in pdf_paths:
        logger.info(
            "开始处理 PDF：file=%s", 
            pdf_path.name,
        )

        try:
            reader = PdfReader(pdf_path)

            logger.info(
                "PDF 打开成功：file=%s page_count=%d", 
                pdf_path.name,
                len(reader.pages),
            )

            for page_index, page in enumerate(reader.pages, start=1):
                try:
                    text = page.extract_text() or ""
                except Exception:
                    logger.exception(
                        "PDF 页面解析失败：file=%s page=%d", 
                        pdf_path.name,
                        page_index,
                    )
                    continue

                clean_text = " ".join(text.split())

                if clean_text:
                    documents.append(
                        {
                            "text": clean_text,
                            "page": page_index,
                            "source": pdf_path.name,
                        }
                    )
                else:
                    logger.debug(
                        "PDF 页面为空：file=%s page=%d", 
                        pdf_path.name,
                        page_index,
                    )
        except Exception:
            logger.exception(
                "无法读取 PDF，已跳过：file=%s", 
                pdf_path.name,
            )

    logger.info(
        "PDF 文本提取完成：document_count=%d", 
        len(documents),
    )

    if not documents:
        raise ValueError(
            "找到了 PDF，但没有成功提取到任何文本。"
        )

    return documents


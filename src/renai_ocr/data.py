from __future__ import annotations

from pathlib import Path
from typing import Iterable

from PIL import Image


def pdf_to_images(pdf_path: str | Path, dpi: int = 300, max_pages: int | None = None):
    pdf_path = Path(pdf_path)

    try:
        from pdf2image import convert_from_path

        pages = convert_from_path(str(pdf_path), dpi=dpi)
        if max_pages is not None:
            pages = pages[:max_pages]
        return pages
    except Exception:
        pass

    import fitz

    doc = fitz.open(str(pdf_path))
    pages = []
    zoom = dpi / 72.0
    matrix = fitz.Matrix(zoom, zoom)

    for i, page in enumerate(doc):
        if max_pages is not None and i >= max_pages:
            break
        pix = page.get_pixmap(matrix=matrix, alpha=False)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        pages.append(img)
    return pages


def extract_pdf_text_pages(pdf_path: str | Path, max_pages: int | None = None) -> list[str]:
    from pypdf import PdfReader

    reader = PdfReader(str(pdf_path))
    texts = []
    for idx, page in enumerate(reader.pages):
        if max_pages is not None and idx >= max_pages:
            break
        try:
            txt = page.extract_text() or ""
        except Exception:
            txt = ""
        texts.append(txt)
    return texts


def list_pdfs(path: str | Path) -> Iterable[Path]:
    return sorted(Path(path).glob("*.pdf"))


def load_text(path: str | Path) -> str:
    return Path(path).read_text(encoding="utf-8", errors="ignore")

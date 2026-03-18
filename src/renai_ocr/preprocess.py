from __future__ import annotations

from typing import Tuple

from PIL import Image


def extract_main_text_region(image: Image.Image, strategy: str = "center_crop", margin: float = 0.08) -> Image.Image:
    if strategy == "none":
        return image

    width, height = image.size
    m_w = int(width * margin)
    m_h = int(height * margin)

    if strategy == "center_crop":
        box: Tuple[int, int, int, int] = (m_w, m_h, width - m_w, height - m_h)
        return image.crop(box)

    raise ValueError(f"Unsupported main text strategy: {strategy}")

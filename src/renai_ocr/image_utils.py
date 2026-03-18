from __future__ import annotations

import base64
import io


def pil_to_base64_png(image) -> str:
    buff = io.BytesIO()
    image.save(buff, format="PNG")
    return base64.b64encode(buff.getvalue()).decode("utf-8")

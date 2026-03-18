from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class OCRBackend(ABC):
    @abstractmethod
    def infer_text(self, image) -> str:
        raise NotImplementedError


class TesseractBackend(OCRBackend):
    def __init__(self, lang: str = "eng"):
        import pytesseract

        self._pt = pytesseract
        self.lang = lang

    def infer_text(self, image) -> str:
        return self._pt.image_to_string(image, lang=self.lang)


class EasyOCRCRNNBackend(OCRBackend):
    def __init__(self, lang_list: list[str] | None = None):
        import easyocr

        self.reader = easyocr.Reader(lang_list or ["en"], gpu=False)

    def infer_text(self, image) -> str:
        arr = np.array(image)
        out = self.reader.readtext(arr, detail=0, paragraph=True)
        return "\n".join(out)


class TrOCRTransformerBackend(OCRBackend):
    def __init__(self, model_name: str = "microsoft/trocr-base-printed"):
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel

        self.processor = TrOCRProcessor.from_pretrained(model_name)
        self.model = VisionEncoderDecoderModel.from_pretrained(model_name)

    def infer_text(self, image) -> str:
        pixel_values = self.processor(images=image, return_tensors="pt").pixel_values
        generated_ids = self.model.generate(pixel_values)
        text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return text.strip()


class EmptyBackend(OCRBackend):
    def infer_text(self, image) -> str:
        return ""


def build_backend(name: str) -> OCRBackend:
    key = name.strip().lower()

    if key in {"tesseract", "tesseract_legacy"}:
        return TesseractBackend()
    if key in {"easyocr", "easyocr_crnn", "crnn"}:
        return EasyOCRCRNNBackend()
    if key in {"trocr", "trocr_transformer", "transformer"}:
        return TrOCRTransformerBackend()
    if key in {"none", "disabled", "pypdf_text"}:
        return EmptyBackend()

    raise ValueError(f"Unsupported OCR backend: {name}")

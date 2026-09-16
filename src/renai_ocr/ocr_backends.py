from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class OCRBackend(ABC):
    @abstractmethod
    def infer_text(self, image) -> str:
        raise NotImplementedError

    def infer_with_confidence(self, image) -> tuple[str, float]:
        """Return (text, confidence) where confidence is in [0.0, 1.0].

        1.0 = high confidence (clean document), 0.0 = very low confidence
        (heavily degraded). Subclasses should override this for real scores;
        the default falls back to infer_text() with a neutral confidence of 1.0.
        """
        return self.infer_text(image), 1.0


class TesseractBackend(OCRBackend):
    def __init__(self, lang: str = "eng"):
        import pytesseract

        self._pt = pytesseract
        self.lang = lang

    def infer_text(self, image) -> str:
        return self._pt.image_to_string(image, lang=self.lang)

    def infer_with_confidence(self, image) -> tuple[str, float]:
        import pandas as pd

        data = self._pt.image_to_data(
            image, lang=self.lang, output_type=self._pt.Output.DATAFRAME
        )
        # Tesseract gives per-word confidence in [-1, 100]; -1 means non-word row.
        word_confs = data.loc[data["conf"] > 0, "conf"]
        conf = float(word_confs.mean()) / 100.0 if not word_confs.empty else 0.0
        text = "\n".join(
            data.loc[data["conf"] > 0, "text"].fillna("").astype(str).tolist()
        )
        return text, conf


class EasyOCRCRNNBackend(OCRBackend):
    def __init__(self, lang_list: list[str] | None = None):
        import easyocr

        self.reader = easyocr.Reader(lang_list or ["en"], gpu=False)

    def infer_text(self, image) -> str:
        arr = np.array(image)
        out = self.reader.readtext(arr, detail=0, paragraph=True)
        return "\n".join(out)

    def infer_with_confidence(self, image) -> tuple[str, float]:
        arr = np.array(image)
        # detail=1 returns list of (bbox, text, prob) tuples
        results = self.reader.readtext(arr, detail=1, paragraph=False)
        if not results:
            return "", 0.0
        texts = [r[1] for r in results]
        probs = [float(r[2]) for r in results]
        conf = float(np.mean(probs))
        return "\n".join(texts), conf


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

    def infer_with_confidence(self, image) -> tuple[str, float]:
        import torch

        pixel_values = self.processor(images=image, return_tensors="pt").pixel_values
        outputs = self.model.generate(
            pixel_values,
            return_dict_in_generate=True,
            output_scores=True,
        )
        generated_ids = outputs.sequences
        text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

        # Compute geometric-mean token confidence from per-step logit scores.
        if outputs.scores:
            token_probs = [
                float(torch.softmax(step_scores, dim=-1).max().item())
                for step_scores in outputs.scores
            ]
            conf = float(np.exp(np.mean(np.log(np.clip(token_probs, 1e-9, 1.0)))))
        else:
            conf = 1.0

        return text, conf


class EmptyBackend(OCRBackend):
    def infer_text(self, image) -> str:
        return ""

    def infer_with_confidence(self, image) -> tuple[str, float]:
        # No OCR was run — signal unknown confidence with 0.0.
        return "", 0.0


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

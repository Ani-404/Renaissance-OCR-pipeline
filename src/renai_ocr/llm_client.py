from __future__ import annotations

import os
import re
from dataclasses import dataclass, field

from openai import OpenAI

# Documents whose OCR confidence falls below this are treated as degraded.
_DEFAULT_CONFIDENCE_THRESHOLD = 0.5


@dataclass
class LLMCleaner:
    api_key_env: str
    model: str
    temperature: float = 0.0
    # Confidence score [0, 1] below which a document is considered degraded /
    # noisy. clean_printed_ocr() will use a more aggressive correction prompt
    # for degraded docs. If also offline, falls back to rule-based cleaning.
    confidence_threshold: float = _DEFAULT_CONFIDENCE_THRESHOLD

    def __post_init__(self):
        api_key = os.getenv(self.api_key_env)
        self.offline_mode = not bool(api_key)
        self.client = OpenAI(api_key=api_key) if api_key else None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _rule_clean(self, text: str) -> str:
        text = re.sub(r"[\t\r]+", " ", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def _is_degraded(self, ocr_confidence: float) -> bool:
        return ocr_confidence < self.confidence_threshold

    def _text_response(self, input_payload) -> str:
        if self.offline_mode:
            for item in input_payload:
                if isinstance(item, dict) and item.get("role") == "user":
                    c = item.get("content")
                    if isinstance(c, str):
                        return self._rule_clean(c)
                    if isinstance(c, list):
                        for part in c:
                            if isinstance(part, dict) and part.get("type") == "input_text":
                                return self._rule_clean(part.get("text", ""))
            return ""

        resp = self.client.responses.create(
            model=self.model,
            temperature=self.temperature,
            input=input_payload,
        )
        return resp.output_text.strip()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def clean_printed_ocr(self, raw_text: str, ocr_confidence: float = 1.0) -> str:
        """Clean OCR output from a printed document.

        Parameters
        ----------
        raw_text:
            Raw OCR text to clean.
        ocr_confidence:
            Mean confidence score from the OCR backend in [0, 1].
            When below ``confidence_threshold``, a stricter, degraded-document
            prompt is used so the LLM knows to be more aggressive about fixing
            errors. When offline AND degraded, falls back to rule-based cleaning
            without an LLM call.
        """
        degraded = self._is_degraded(ocr_confidence)

        # Offline + degraded: rule-based is better than returning raw garbage.
        if self.offline_mode:
            return self._rule_clean(raw_text)

        if degraded:
            system_prompt = (
                "You are correcting OCR from a highly degraded or noisy early modern "
                "printed document. The OCR quality is low (confidence {:.0%}). "
                "Be aggressive: fix character substitutions, run-on tokens, ligature "
                "errors, and OCR artefacts. Reconstruct plausible historical text where "
                "the OCR is clearly wrong. Preserve historical spelling where plausible "
                "and return plain corrected text only.".format(ocr_confidence)
            )
        else:
            system_prompt = (
                "You are correcting OCR from early modern printed text. "
                "Preserve historical spelling whenever plausible, but fix obvious OCR "
                "noise, split tokens, ligature errors, and character substitutions. "
                "Return plain corrected text only."
            )

        return self._text_response(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": raw_text},
            ]
        )

    def analyze_handwritten_page(self, image_b64: str) -> str:
        if self.offline_mode:
            return "- offline mode\n- using OCR prior driven transcription"
        prompt = (
            "Analyze this handwritten page before transcription. "
            "Briefly describe script characteristics, likely abbreviations, and line flow cues. "
            "Return a short bullet list as plain text."
        )
        return self._text_response(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": prompt},
                        {"type": "input_image", "image_url": f"data:image/png;base64,{image_b64}"},
                    ],
                }
            ]
        )

    def transcribe_handwriting_page(self, image_b64: str, page_analysis: str = "", ocr_prior: str = "") -> str:
        if self.offline_mode:
            return self._rule_clean(ocr_prior)
        prompt = (
            "Transcribe this handwritten page faithfully. "
            "Use the analysis hints and OCR prior if useful, but prefer what is visible on page. "
            "Keep historical spelling, line breaks where feasible, and output plain text only."
        )
        user_text = (
            f"Page analysis hints:\n{page_analysis or '[none]'}\n\n"
            f"OCR prior:\n{ocr_prior or '[none]'}\n"
        )
        return self._text_response(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": prompt + "\n\n" + user_text},
                        {"type": "input_image", "image_url": f"data:image/png;base64,{image_b64}"},
                    ],
                }
            ]
        )

    def correct_handwriting_text(self, draft_text: str, prior_context: str = "") -> str:
        if self.offline_mode:
            return self._rule_clean(draft_text)
        prompt = (
            "You are refining a handwritten transcription. "
            "Correct only likely recognition errors while preserving original orthography and meaning. "
            "Use prior context if provided for coherence. Return plain text only."
        )
        user_text = f"Prior context:\n{prior_context or '[none]'}\n\nDraft:\n{draft_text}"
        return self._text_response(
            [
                {"role": "system", "content": prompt},
                {"role": "user", "content": user_text},
            ]
        )

    def finalize_handwritten_source(self, source_text: str) -> str:
        if self.offline_mode:
            return self._rule_clean(source_text)
        prompt = (
            "Finalize this source-level handwritten transcription. "
            "Remove duplicated artifacts, keep spelling historical, preserve readability, "
            "and return plain text only."
        )
        return self._text_response(
            [
                {"role": "system", "content": prompt},
                {"role": "user", "content": source_text},
            ]
        )

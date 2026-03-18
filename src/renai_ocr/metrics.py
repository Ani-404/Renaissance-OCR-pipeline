from __future__ import annotations

import re
from typing import Iterable

import numpy as np


def normalize_for_eval(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def levenshtein_distance(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)

    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i]
        for j, cb in enumerate(b, start=1):
            cost = 0 if ca == cb else 1
            cur.append(min(cur[-1] + 1, prev[j] + 1, prev[j - 1] + cost))
        prev = cur
    return prev[-1]


def cer(reference: str, hypothesis: str, normalize: bool = False) -> float:
    ref = normalize_for_eval(reference) if normalize else reference.strip()
    hyp = normalize_for_eval(hypothesis) if normalize else hypothesis.strip()
    if not ref:
        return 0.0 if not hyp else 1.0
    return levenshtein_distance(ref, hyp) / max(1, len(ref))


def wer(reference: str, hypothesis: str, normalize: bool = False) -> float:
    ref_text = normalize_for_eval(reference) if normalize else reference.strip()
    hyp_text = normalize_for_eval(hypothesis) if normalize else hypothesis.strip()
    ref_words = ref_text.split()
    hyp_words = hyp_text.split()
    if not ref_words:
        return 0.0 if not hyp_words else 1.0
    ref = "\n".join(ref_words)
    hyp = "\n".join(hyp_words)
    return levenshtein_distance(ref, hyp) / len(ref_words)


def exact_match_rate(refs: Iterable[str], hyps: Iterable[str]) -> float:
    refs = list(refs)
    hyps = list(hyps)
    if not refs:
        return 0.0
    matched = sum(int(r.strip() == h.strip()) for r, h in zip(refs, hyps))
    return matched / len(refs)


def summarize_scores(cers: list[float], wers: list[float], ncers: list[float], nwers: list[float]) -> dict:
    return {
        "cer_mean": float(np.mean(cers)) if cers else None,
        "cer_median": float(np.median(cers)) if cers else None,
        "wer_mean": float(np.mean(wers)) if wers else None,
        "wer_median": float(np.median(wers)) if wers else None,
        "normalized_cer_mean": float(np.mean(ncers)) if ncers else None,
        "normalized_wer_mean": float(np.mean(nwers)) if nwers else None,
    }

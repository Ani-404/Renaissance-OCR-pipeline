from __future__ import annotations

import json
from pathlib import Path

from tqdm import tqdm

from .config import LLMConfig, PipelineConfig
from .data import extract_pdf_text_pages, list_pdfs, load_text, pdf_to_images
from .image_utils import pil_to_base64_png
from .llm_client import LLMCleaner
from .metrics import cer, summarize_scores, wer
from .ocr_backends import build_backend
from .preprocess import extract_main_text_region


def _gt_path(gt_dir: Path, pdf_path: Path) -> Path:
    return gt_dir / f"{pdf_path.stem}.txt"


def _dump_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def run_test1(cfg: PipelineConfig, llm_cfg: LLMConfig) -> dict:
    cfg.out_dir.mkdir(parents=True, exist_ok=True)

    ocr = build_backend(cfg.ocr_backend)
    cleaner = LLMCleaner(llm_cfg.api_key_env, llm_cfg.model, llm_cfg.temperature) if cfg.use_llm else None

    cers, wers, ncers, nwers = [], [], [], []
    records = []

    for pdf in tqdm(list_pdfs(cfg.pdf_dir), desc="Test I PDFs"):
        if cfg.ocr_backend.lower() == "pypdf_text":
            page_predictions = extract_pdf_text_pages(pdf, max_pages=cfg.max_pages)
            pages_len = len(page_predictions)
        else:
            pages = pdf_to_images(pdf, dpi=cfg.dpi, max_pages=cfg.max_pages)
            page_predictions = []
            for idx, page in enumerate(pages, start=1):
                processed = extract_main_text_region(page, cfg.main_text_strategy, cfg.main_text_margin)
                raw = ocr.infer_text(processed)
                page_predictions.append(raw)
                if cfg.save_page_outputs:
                    (cfg.out_dir / f"{pdf.stem}.page_{idx:03d}.raw.txt").write_text(raw, encoding="utf-8")
            pages_len = len(pages)

        raw_text = "\n".join(page_predictions)
        final_text = cleaner.clean_printed_ocr(raw_text) if cleaner else raw_text

        gt_file = _gt_path(cfg.gt_dir, pdf)
        if gt_file.exists():
            gt = load_text(gt_file)
            c = cer(gt, final_text)
            w = wer(gt, final_text)
            nc = cer(gt, final_text, normalize=True)
            nw = wer(gt, final_text, normalize=True)
            cers.append(c)
            wers.append(w)
            ncers.append(nc)
            nwers.append(nw)
        else:
            c, w, nc, nw = None, None, None, None

        rec = {
            "source": pdf.name,
            "num_pages": pages_len,
            "chars": len(final_text),
            "cer": c,
            "wer": w,
            "normalized_cer": nc,
            "normalized_wer": nw,
        }
        records.append(rec)

        (cfg.out_dir / f"{pdf.stem}.pred.txt").write_text(final_text, encoding="utf-8")

    summary = summarize_scores(cers, wers, ncers, nwers)
    payload = {
        "task": "test1",
        "model": cfg.model_name,
        "ocr_backend": cfg.ocr_backend,
        "llm_enabled": cfg.use_llm,
        "summary": summary,
        "records": records,
    }
    _dump_json(cfg.out_dir / "metrics.test1.json", payload)
    return payload


def run_test2(cfg: PipelineConfig, llm_cfg: LLMConfig) -> dict:
    cfg.out_dir.mkdir(parents=True, exist_ok=True)

    cleaner = LLMCleaner(llm_cfg.api_key_env, llm_cfg.model, llm_cfg.temperature)
    ocr = None
    if cfg.use_ocr_prior and cfg.ocr_backend.lower() not in {"none", "disabled"}:
        ocr = build_backend(cfg.ocr_backend)

    cers, wers, ncers, nwers = [], [], [], []
    records = []

    for pdf in tqdm(list_pdfs(cfg.pdf_dir), desc="Test II PDFs"):
        pages = pdf_to_images(pdf, dpi=cfg.dpi, max_pages=cfg.max_pages)
        pdf_text_prior = extract_pdf_text_pages(pdf, max_pages=cfg.max_pages) if cfg.ocr_backend.lower() == "pypdf_text" else []

        page_texts = []
        for idx, page in enumerate(pages, start=1):
            processed = extract_main_text_region(page, cfg.main_text_strategy, cfg.main_text_margin)
            encoded = pil_to_base64_png(processed)

            page_analysis = cleaner.analyze_handwritten_page(encoded) if cfg.llm_every_stage else ""
            if cfg.ocr_backend.lower() == "pypdf_text":
                ocr_prior = pdf_text_prior[idx - 1] if idx - 1 < len(pdf_text_prior) else ""
            else:
                ocr_prior = ocr.infer_text(processed) if ocr else ""

            page_draft = cleaner.transcribe_handwriting_page(encoded, page_analysis=page_analysis, ocr_prior=ocr_prior)
            context = "\n".join(page_texts[-2:]) if page_texts else ""
            page_final = cleaner.correct_handwriting_text(page_draft, prior_context=context)
            page_texts.append(page_final)

            if cfg.save_page_outputs:
                stage_text = (
                    f"# analysis\n{page_analysis}\n\n"
                    f"# ocr_prior\n{ocr_prior}\n\n"
                    f"# page_final\n{page_final}\n"
                )
                (cfg.out_dir / f"{pdf.stem}.page_{idx:03d}.stages.txt").write_text(stage_text, encoding="utf-8")

        source_draft = "\n".join(page_texts)
        final_text = cleaner.finalize_handwritten_source(source_draft)

        gt_file = _gt_path(cfg.gt_dir, pdf)
        if gt_file.exists():
            gt = load_text(gt_file)
            c = cer(gt, final_text)
            w = wer(gt, final_text)
            nc = cer(gt, final_text, normalize=True)
            nw = wer(gt, final_text, normalize=True)
            cers.append(c)
            wers.append(w)
            ncers.append(nc)
            nwers.append(nw)
        else:
            c, w, nc, nw = None, None, None, None

        rec = {
            "source": pdf.name,
            "num_pages": len(pages),
            "chars": len(final_text),
            "cer": c,
            "wer": w,
            "normalized_cer": nc,
            "normalized_wer": nw,
            "ocr_prior_used": bool(cfg.use_ocr_prior),
            "llm_every_stage": cfg.llm_every_stage,
        }
        records.append(rec)

        (cfg.out_dir / f"{pdf.stem}.pred.txt").write_text(final_text, encoding="utf-8")

    summary = summarize_scores(cers, wers, ncers, nwers)
    payload = {
        "task": "test2",
        "model": cfg.model_name,
        "ocr_backend": cfg.ocr_backend,
        "llm_enabled": True,
        "summary": summary,
        "records": records,
    }
    _dump_json(cfg.out_dir / "metrics.test2.json", payload)
    return payload

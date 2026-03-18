# Deliverable Matrix

## Fully Implemented in Code

- Test I OCR pipeline with selectable architecture (`easyocr_crnn` and `trocr_transformer`).
- Main-text focus step that crops margins to reduce marginalia noise.
- Late-stage LLM cleanup for printed OCR.
- Test II LLM/VLM-centric multi-stage pipeline (analysis -> transcription -> correction -> finalization).
- Optional OCR prior integration in Test II.
- CER/WER + normalized CER/WER evaluation.
- Per-source and per-page/stage output logging.
- Ablation runner for comparative evaluation.
- Notebook generator for submission artifact creation.
- Report template and strategy notes.

## Requires Your Local Run to Finalize Submission

- Running pipelines on your selected dataset PDFs.
- Generating final metrics files in `outputs/`.
- Generating notebook with real outputs and exporting to PDF.
- Pushing your branch and emailing final links + CV.

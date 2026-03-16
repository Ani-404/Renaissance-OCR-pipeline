# RenAIssance OCR Pipeline (GSoC 2026 Test Scaffold)

This repository provides a practical, reproducible baseline for both RenAIssance tests:

- `Test I` (printed OCR): OCR model + late-stage LLM cleanup.
- `Test II` (handwritten OCR): LLM/VLM-driven OCR pipeline used throughout recognition and correction.

## Project Structure

- `src/renai_ocr/` core code
- `scripts/` runnable entrypoints
- `configs/` sample YAML configs
- `docs/` report template + methodology notes

## Quick Start

1. Create environment and install dependencies:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .
```

2. Put your dataset under:

```text
data/
  printed/
    source_01.pdf
    ...
  handwritten/
    source_01.pdf
    ...
  ground_truth/
    source_01.txt
    ...
```

3. Run Test I baseline:

```powershell
python scripts/run_test1.py --config configs/test1.sample.yaml
```

4. Run Test II baseline:

```powershell
python scripts/run_test2.py --config configs/test2.sample.yaml
```

## What This Covers

- PDF page extraction and normalization
- OCR inference abstraction (EasyOCR/Tesseract-compatible interfaces)
- VLM/LLM correction stage
- CER/WER + exact-match evaluation
- Structured output for notebook/report export

## Submission Checklist

- Keep work in your own branch (no PR).
- Export notebook with outputs and PDF.
- Send CV + repo link to `human-ai@cern.ch` with subject:
  `Evaluation Test: RenAIssance`.

# RenAIssance OCR Pipeline (GSoC 2026)

This repository is structured to satisfy both official evaluation tests:

- Test I: printed OCR with OCR architecture + late-stage LLM/VLM cleanup.
- Test II: handwritten OCR pipeline where LLM/VLM is used throughout all stages.

## Requirement Mapping

### Test I (Printed)
- OCR architecture options implemented:
  - convolutional-recurrent: `easyocr_crnn`
  - transformer: `trocr_transformer`
- Main-text extraction (marginalia suppression):
  - `main_text_strategy: center_crop`
- Late-stage LLM cleanup:
  - `LLMCleaner.clean_printed_ocr(...)`
- Metrics:
  - CER, WER, normalized CER/WER, per-source JSON results.

### Test II (Handwritten)
- LLM/VLM used throughout pipeline stages:
  - page analysis: `analyze_handwritten_page(...)`
  - page transcription: `transcribe_handwriting_page(...)`
  - page correction: `correct_handwriting_text(...)`
  - source finalization: `finalize_handwritten_source(...)`
- Optional OCR integration:
  - `use_ocr_prior: true` with `easyocr_crnn`
- Metrics:
  - CER, WER, normalized CER/WER, plus stage traces.

## Repository Layout

- `src/renai_ocr/` core package
- `scripts/run_test1.py` run Test I
- `scripts/run_test2.py` run Test II
- `scripts/run_all_ablation.py` run all baseline/ablation configs
- `scripts/finetune_trocr.py` transformer fine-tuning utility (CSV image-text pairs)
- `scripts/build_notebook.py` generate `.ipynb` summary from metrics files
- `configs/` test and ablation configs
- `docs/` methodology and report template

## Setup

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .
```

## Data Layout

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

Ground-truth filenames must match PDF stem names.

## Run Test I

```powershell
python scripts/run_test1.py --config configs/test1.sample.yaml
```

## Run Test II

```powershell
python scripts/run_test2.py --config configs/test2.sample.yaml
```

## Run Ablations

```powershell
python scripts/run_all_ablation.py
```

Default ablations:
- `configs/test1.ocr_only.yaml`
- `configs/test1.transformer_plus_llm.yaml`
- `configs/test2.vlm_only.yaml`
- `configs/test2.vlm_plus_ocr.yaml`

## Generate Notebook Artifact

```powershell
python scripts/build_notebook.py --metrics \
  outputs/test1/metrics.test1.json \
  outputs/test2/metrics.test2.json \
  --out notebooks/renaissance_submission.ipynb
```

Export notebook to PDF from Jupyter UI for final submission package.

## Suggested Branch Flow (No PR)

```powershell
git checkout -b codex/renaissance-gsoc-2026
# commit your work
# push your branch
```

## Submission Packet

Send to `human-ai@cern.ch` with title `Evaluation Test: RenAIssance`:
- CV
- GitHub repo/branch link
- Jupyter notebook (`.ipynb`) with outputs
- PDF export of notebook with outputs

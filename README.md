# RenAIssance OCR Pipeline

OCR system for early modern (Renaissance-era) documents — both **printed** and **handwritten** — built for GSoC 2026 evaluation.

---

## How it works (core idea)

Every PDF page goes through three layers, in order:

```
PDF page
   │
   ▼
[1] Preprocess       — crop away margins to suppress marginalia
   │
   ▼
[2] OCR / VLM        — extract raw text (+ a confidence score per page)
   │
   ▼
[3] LLM cleanup      — fix errors; uses a stricter prompt when confidence is low
```

The **confidence score** (0–1) is what makes the fallback real:

| OCR confidence | Online (API key set) | Offline (no API key) |
|---|---|---|
| ≥ threshold (clean doc) | Standard LLM correction prompt | Rule-based whitespace clean |
| < threshold (degraded doc) | **Aggressive** LLM correction prompt | Rule-based whitespace clean |

Threshold default: **0.5**. Tune it per-config with `confidence_threshold`.

---

## Two pipelines

### Test I — Printed documents

```
Page image → EasyOCR (CRNN) or TrOCR (Transformer)
           → mean page confidence computed
           → LLMCleaner.clean_printed_ocr(text, ocr_confidence)
           → CER / WER scored against ground truth
```

Switch backend in config: `ocr_backend: easyocr_crnn` or `trocr_transformer`.

### Test II — Handwritten documents

VLM (GPT-4 Vision) does all the heavy lifting across **4 sequential stages per page**:

```
Page image
  │
  ├─ [Stage 1] analyze_handwritten_page()     — describe script, abbreviations, layout
  │
  ├─ [Stage 2] transcribe_handwriting_page()  — raw transcription (uses stage 1 + optional OCR prior)
  │
  ├─ [Stage 3] correct_handwriting_text()     — coherence pass using previous 2 pages as context
  │
  └─ [Stage 4] finalize_handwritten_source()  — whole-document deduplication & polish
```

Each stage's output is saved to `outputs/<name>.page_NNN.stages.txt` for tracing.

---

## OCR backends and their confidence scores

| Key in config | Model | Confidence source |
|---|---|---|
| `easyocr_crnn` | EasyOCR (CRNN) | Mean of per-detection-box probability |
| `trocr_transformer` | Microsoft TrOCR | Geometric mean of per-token softmax peak |
| `tesseract` | Tesseract | Mean of per-word Tesseract conf column |
| `pypdf_text` | pdfminer (native text) | Always 1.0 (clean by definition) |

---

## Metrics

Computed per source document and averaged across the run:

- **CER** — character error rate (Levenshtein / ref length)
- **WER** — word error rate (same idea, word-level)
- **Normalized CER/WER** — lowercased + collapsed whitespace before scoring

Results written to `outputs/<test>/metrics.<test>.json`. Each record includes `ocr_confidence` and a `degraded: true/false` flag.

---

## Setup

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .
$env:OPENAI_API_KEY = "sk-..."   # omit to run fully offline (rule-based fallback)
```

## Data layout

```
data/
  printed/          <- PDFs for Test I
  handwritten/      <- PDFs for Test II
  ground_truth/     <- <stem>.txt files matching PDF names
```

## Run

```powershell
# Test I (printed)
python scripts/run_test1.py --config configs/test1.sample.yaml

# Test II (handwritten)
python scripts/run_test2.py --config configs/test2.sample.yaml

# All 6 ablation configs in one go
python scripts/run_all_ablation.py
```

## Ablation configs

| Config file | What it tests |
|---|---|
| `test1.ocr_only.yaml` | CRNN only, no LLM |
| `test1.sample.yaml` | CRNN + LLM cleanup |
| `test1.transformer_plus_llm.yaml` | TrOCR + LLM cleanup |
| `test2.sample.yaml` | Full VLM pipeline + OCR prior |
| `test2.vlm_only.yaml` | VLM pipeline, no OCR prior |
| `test2.vlm_plus_ocr.yaml` | VLM pipeline + EasyOCR prior |

## Generate submission notebook

```powershell
python scripts/build_notebook.py ^
  --metrics outputs/test1/metrics.test1.json outputs/test2/metrics.test2.json ^
  --out notebooks/renaissance_submission.ipynb
```

Export to PDF from Jupyter for the final submission packet.

---

## Repository layout

```
src/renai_ocr/
  pipeline.py        <- run_test1() and run_test2() orchestration
  ocr_backends.py    <- EasyOCR, TrOCR, Tesseract — each returns (text, confidence)
  llm_client.py      <- LLMCleaner — 4 VLM methods + confidence-aware printed cleanup
  metrics.py         <- CER, WER, normalized variants, summary stats
  preprocess.py      <- margin cropping (center_crop strategy)
  config.py          <- YAML -> dataclass loader
  data.py            <- PDF -> images, PDF text extraction, ground-truth loader
  image_utils.py     <- PIL -> base64 PNG for VLM calls
scripts/
  run_test1.py       <- CLI entry for Test I
  run_test2.py       <- CLI entry for Test II
  run_all_ablation.py
  finetune_trocr.py  <- fine-tune TrOCR on CSV image-text pairs
  build_notebook.py  <- stitch metrics JSON into .ipynb
configs/             <- 6 YAML configs (see ablation table above)
```

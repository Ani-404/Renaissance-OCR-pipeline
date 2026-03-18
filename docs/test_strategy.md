# RenAIssance Test Strategy

## Test I (Printed Sources)

Approach:
1. Convert PDF pages to images (`300 DPI`).
2. Apply line-level OCR (EasyOCR or Tesseract).
3. Concatenate page text in reading order.
4. Late-stage LLM cleanup:
   - remove OCR noise
   - fix obvious segmentation errors
   - preserve likely historical spelling
5. Evaluate against provided transcription.

Why this matches requirements:
- Core recognizer is OCR (convolutional or transformer-based backend can be swapped).
- LLM is explicitly late-stage.

## Test II (Handwritten Sources)

Approach:
1. Convert PDF pages to images.
2. Use VLM/LLM directly for page transcription (image input + reasoning prompt).
3. Optionally add OCR prior and ask the LLM to reconcile variants.
4. Aggregate pages into final source-level transcription.
5. Evaluate with CER/WER and qualitative error analysis.

Why this matches requirements:
- LLM/VLM is used at all recognition stages, not only post-processing.

## Evaluation Metrics

Primary:
- CER (Character Error Rate): robust for historical orthography and character-level OCR noise.
- WER (Word Error Rate): captures readability and lexical correctness.

Secondary:
- Exact Match Rate on aligned text chunks.
- Error type distribution (substitution/insertion/deletion, optional).

Reporting:
- Per-source CER/WER.
- Macro averages across all sources.
- Before vs after cleanup (for Test I).
- Ablation (OCR only vs OCR+LLM, or VLM only vs VLM+OCR prior).

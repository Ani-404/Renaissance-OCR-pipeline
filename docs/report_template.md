# Evaluation Report Template

## 1. Objective
State whether this submission targets Test I, Test II, or both.

## 2. Dataset
- Number of PDF sources used
- Printed or handwritten subset
- Train/validation/test split or source-level holdout strategy

## 3. Method
- OCR / VLM architecture
- Prompting strategy
- Preprocessing decisions (denoise, contrast, binarization)
- Post-processing decisions

## 4. Metrics
- CER
- WER
- (Optional) exact match rate

## 5. Results
| Source | CER | WER | Notes |
|---|---:|---:|---|
| source_01 |  |  |  |

Include macro-average and median.

## 6. Error Analysis
- Common failure modes
- Examples of corrected OCR outputs
- Remaining limitations

## 7. Discussion
- Why this approach is appropriate for early modern documents
- Next steps for improving robustness and scaling

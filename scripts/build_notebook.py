from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_notebook(metrics_paths: list[Path], out_path: Path) -> None:
    cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# RenAIssance Evaluation Notebook\n",
                "This notebook summarizes OCR/VLM experiment outputs for GSoC test submission."
            ],
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Loaded Metrics Files\n",
                "- " + "\n- ".join(str(p) for p in metrics_paths),
            ],
        },
    ]

    for path in metrics_paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        cells.append(
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [f"## {path.name}\n", "```json\n", json.dumps(payload.get("summary", {}), indent=2), "\n```"],
            }
        )

    cells.append(
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Interpretation\n",
                "- Compare CER/WER across ablations.\n",
                "- Highlight where LLM/VLM improves OCR robustness.\n",
                "- Discuss remaining failure modes (layout drift, historical spelling ambiguity).",
            ],
        }
    )

    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(notebook, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a submission notebook from metrics files")
    parser.add_argument("--metrics", nargs="+", required=True)
    parser.add_argument("--out", default="notebooks/renaissance_submission.ipynb")
    args = parser.parse_args()

    metrics_paths = [Path(p) for p in args.metrics]
    build_notebook(metrics_paths, Path(args.out))
    print(f"Notebook written to {args.out}")


if __name__ == "__main__":
    main()

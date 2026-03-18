from __future__ import annotations

import argparse
import json
from pathlib import Path

from renai_ocr.config import load_pipeline_config
from renai_ocr.pipeline import run_test1, run_test2


def _save(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run all RenAIssance ablations")
    parser.add_argument(
        "--configs",
        nargs="+",
        default=[
            "configs/test1.ocr_only.yaml",
            "configs/test1.transformer_plus_llm.yaml",
            "configs/test2.vlm_only.yaml",
            "configs/test2.vlm_plus_ocr.yaml",
        ],
    )
    parser.add_argument("--out", default="outputs/ablation_summary.json")
    args = parser.parse_args()

    runs = []
    for cfg_path in args.configs:
        cfg, llm_cfg = load_pipeline_config(cfg_path)
        if cfg.task == "test1":
            payload = run_test1(cfg, llm_cfg)
        elif cfg.task == "test2":
            payload = run_test2(cfg, llm_cfg)
        else:
            raise ValueError(f"Unsupported task: {cfg.task}")
        runs.append({"config": cfg_path, "result": payload})

    _save(Path(args.out), {"runs": runs})
    print(json.dumps({"num_runs": len(runs), "out": args.out}, indent=2))


if __name__ == "__main__":
    main()

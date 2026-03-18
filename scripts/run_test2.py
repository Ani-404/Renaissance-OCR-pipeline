from __future__ import annotations

import argparse
import json

from renai_ocr.config import load_pipeline_config
from renai_ocr.pipeline import run_test2


def main() -> None:
    parser = argparse.ArgumentParser(description="Run RenAIssance Test II pipeline")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    args = parser.parse_args()

    cfg, llm_cfg = load_pipeline_config(args.config)
    payload = run_test2(cfg, llm_cfg)
    print(json.dumps(payload["summary"], indent=2))


if __name__ == "__main__":
    main()

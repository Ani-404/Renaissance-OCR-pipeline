from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import yaml


@dataclass
class PipelineConfig:
    data_dir: Path
    pdf_dir: Path
    gt_dir: Path
    out_dir: Path
    task: str
    ocr_backend: str
    use_llm: bool
    model_name: str
    max_pages: int | None = None
    dpi: int = 300
    main_text_strategy: str = "center_crop"
    main_text_margin: float = 0.08
    llm_every_stage: bool = False
    use_ocr_prior: bool = False
    save_page_outputs: bool = True


@dataclass
class LLMConfig:
    api_key_env: str
    model: str
    temperature: float = 0.0


def read_yaml(path: str | Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_pipeline_config(path: str | Path) -> tuple[PipelineConfig, LLMConfig]:
    raw = read_yaml(path)
    pipe = raw["pipeline"]
    llm = raw.get("llm", {})

    cfg = PipelineConfig(
        data_dir=Path(pipe["data_dir"]),
        pdf_dir=Path(pipe["pdf_dir"]),
        gt_dir=Path(pipe["gt_dir"]),
        out_dir=Path(pipe["out_dir"]),
        task=pipe.get("task", "test1"),
        ocr_backend=pipe.get("ocr_backend", "easyocr_crnn"),
        use_llm=bool(pipe.get("use_llm", True)),
        model_name=pipe.get("model_name", "baseline"),
        max_pages=pipe.get("max_pages"),
        dpi=int(pipe.get("dpi", 300)),
        main_text_strategy=pipe.get("main_text_strategy", "center_crop"),
        main_text_margin=float(pipe.get("main_text_margin", 0.08)),
        llm_every_stage=bool(pipe.get("llm_every_stage", False)),
        use_ocr_prior=bool(pipe.get("use_ocr_prior", False)),
        save_page_outputs=bool(pipe.get("save_page_outputs", True)),
    )

    llm_cfg = LLMConfig(
        api_key_env=llm.get("api_key_env", "OPENAI_API_KEY"),
        model=llm.get("model", "gpt-4.1-mini"),
        temperature=float(llm.get("temperature", 0.0)),
    )
    return cfg, llm_cfg

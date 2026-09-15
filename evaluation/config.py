"""Validated experiment configuration used by Python, YAML, and Jupyter."""

from __future__ import annotations
from pathlib import Path
from typing import Literal
import yaml
from pydantic import BaseModel, Field

class SingleSystemConfig(BaseModel):
    name: str
    provider: Literal["openai", "anthropic"]
    model: str
    enabled: bool = True

class MultiAgentModels(BaseModel):
    supervisor: str
    specialist: str
    judge: str
    safety: str

class MetricsConfig(BaseModel):
    bootstrap_samples: int = 2000
    confidence_level: float = Field(default=.95, gt=0, lt=1)
    positive_label: str | None = None

class ExperimentConfig(BaseModel):
    name: str = "multi-vs-single"
    dataset: str = "data/test_cases.jsonl"
    output_dir: str = "results/experiment"
    task_type: Literal["qa", "binary_classification", "multiclass_classification"] = "qa"
    temperature: float = 0
    repetitions: int = Field(default=1, ge=1)
    resume: bool = True
    multi_agent_enabled: bool = True
    multi_agent_provider: Literal["openai"] = "openai"
    multi_agent_models: MultiAgentModels
    single_agents: list[SingleSystemConfig]
    metrics: MetricsConfig = Field(default_factory=MetricsConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ExperimentConfig":
        return cls.model_validate(yaml.safe_load(Path(path).read_text(encoding="utf-8")))

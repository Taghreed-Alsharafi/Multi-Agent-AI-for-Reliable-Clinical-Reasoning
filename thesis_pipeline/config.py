from __future__ import annotations
from dataclasses import asdict, dataclass, field
import hashlib, json
from pathlib import Path

@dataclass
class ThesisConfig:
    dataset: dict
    run_mode: str = "test"
    global_seed: int = 42
    gpt_model: str = "gpt-5-mini"
    claude_model: str = "claude-haiku-4-5-20251001"
    output_dir: str = "results/thesis"
    pilot_size: int = 25
    test_size: int = 5
    n_agents: int = 4
    bootstrap_samples: int = 2000
    prompt_versions: dict = field(default_factory=lambda: {"single": "mcq-v1", "multi": "mcq-v1"})
    report_only: bool = False
    mock: bool = False

    def resolved_output(self, root: str | Path = ".") -> Path:
        p = Path(self.output_dir)
        return p if p.is_absolute() else Path(root) / p

    @property
    def configuration_hash(self) -> str:
        payload = asdict(self); payload.pop("report_only", None)
        return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()

"""Reproducible thesis experiment controller and backend utilities."""

from .config import ThesisConfig
from .data import DatasetBundle, initialize_dataset
from .pipeline import ThesisPipeline, initialize_thesis
def _current():
    from .pipeline import _CURRENT
    if _CURRENT is None: raise RuntimeError("Call initialize_thesis first")
    return _CURRENT
def run_single_baselines(): return _current().run_single_baselines()
def run_agreement_experiments(): return _current().run_agreement_experiments()
def select_best_agreement_method(): return _current().select_best_agreement_method()
def run_final_comparison(): return _current().run_final_comparison()
def build_thesis_outputs(): return _current().build_thesis_outputs()

__all__ = ["DatasetBundle", "ThesisConfig", "ThesisPipeline", "initialize_dataset", "initialize_thesis", "run_single_baselines", "run_agreement_experiments", "select_best_agreement_method", "run_final_comparison", "build_thesis_outputs"]

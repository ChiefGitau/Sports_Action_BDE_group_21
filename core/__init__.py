"""Core: shared config, data contract, model interface, metrics, cost tools and file formats.

Every other module imports from here and communicates through the saved files defined in
core.artifacts. Only the Core owner changes these formats.
"""
from .artifacts import (
    RESULT_COLUMNS,
    ResultRow,
    append_result,
    load_flops,
    load_logits,
    load_results,
    load_thresholds,
    save_flops,
    save_logits,
    save_thresholds,
)
from .config import REPO_ROOT, AttrDict, load_config, resolve_path, stride_frames, window_frames
from .data import SPLITS, class_balance, load_split, make_loader, save_split, validate_split
from .flops import FLOPS_TOOL, flops_per_exit, param_count, profile, state_dict_size_mb
from .metrics import compute_metrics, confusion_matrix
from .mock import make_mock_split, write_mock_dataset
from .model import DummyExitModel, ExitModel, stack_exit_logits
from .seed import set_seed

__all__ = [
    "RESULT_COLUMNS", "ResultRow", "append_result", "load_flops", "load_logits", "load_results",
    "load_thresholds", "save_flops", "save_logits", "save_thresholds",
    "REPO_ROOT", "AttrDict", "load_config", "resolve_path", "stride_frames", "window_frames",
    "SPLITS", "class_balance", "load_split", "make_loader", "save_split", "validate_split",
    "FLOPS_TOOL", "flops_per_exit", "param_count", "profile", "state_dict_size_mb",
    "compute_metrics", "confusion_matrix",
    "make_mock_split", "write_mock_dataset",
    "DummyExitModel", "ExitModel", "stack_exit_logits",
    "set_seed",
]

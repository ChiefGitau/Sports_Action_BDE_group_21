"""M2 static baselines and M3 early-exit model. Both implement core.ExitModel on one shared
temporal CNN trunk (models/cnn.py) and communicate with other modules only through the files
written by models.train.export. See adaptive_inference_project_plan.md.
"""
from .baseline import StaticCNN
from .cnn import ExitCNN, featurize
from .early_exit import EarlyExitCNN
from .prune import prune_channels
from .registry import KINDS, arch_from_config, build_model, load_checkpoint, save_checkpoint
from .train import evaluate, exit_loss, export, fit, get_device, load_splits, predict

__all__ = [
    "StaticCNN", "ExitCNN", "featurize", "EarlyExitCNN", "prune_channels",
    "KINDS", "arch_from_config", "build_model", "load_checkpoint", "save_checkpoint",
    "evaluate", "exit_loss", "export", "fit", "get_device", "load_splits", "predict",
]

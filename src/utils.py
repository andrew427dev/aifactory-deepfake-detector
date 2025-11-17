"""Utility helpers shared across training scripts."""
from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import torch
from tqdm import tqdm
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def set_seed(seed: int | None) -> None:
    """Seed Python, NumPy, and PyTorch RNGs for reproducibility."""
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, 'cudnn'):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


def get_device(preferred: str | None = None) -> torch.device:
    """Return best available device (preferring MPS on macOS, then CUDA)."""
    if preferred:
        preferred = preferred.lower()
        if preferred == 'cuda' and torch.cuda.is_available():
            return torch.device('cuda')
        if preferred == 'mps' and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return torch.device('mps')
        return torch.device(preferred)

    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def save_model(model: torch.nn.Module, path: str | os.PathLike[str]) -> None:
    """Persist a PyTorch model's state dict to ``path`` (directories auto-created)."""
    path = Path(path)
    if path.parent:
        path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)


def load_model(
    model: torch.nn.Module,
    path: str | os.PathLike[str],
    device: torch.device | None = None,
) -> torch.nn.Module:
    """Load model weights from ``path`` and return the model instance."""
    map_location = device or get_device()
    state_dict = torch.load(path, map_location=map_location)
    model.load_state_dict(state_dict)
    return model.to(map_location)


def to_absolute_path(path: str | os.PathLike[str] | None) -> str | None:
    """Resolve ``path`` relative to the project root when not absolute."""
    if path is None:
        return None
    path_str = os.fspath(path)
    if not path_str:
        return None
    if os.path.isabs(path_str):
        return path_str
    return os.path.join(str(PROJECT_ROOT), path_str)


def load_yaml_config(config_path: str | os.PathLike[str] | None = 'config.yaml') -> dict:
    """Load and parse a YAML configuration file relative to the repo root."""
    resolved_path = to_absolute_path(config_path or 'config.yaml')
    if resolved_path is None:
        raise ValueError('A configuration path must be provided.')
    with open(resolved_path, 'r', encoding='utf-8') as config_file:
        return yaml.safe_load(config_file) or {}


def compute_accuracy(predictions: Sequence[float], targets: Sequence[float], threshold: float = 0.5) -> float:
    if not predictions:
        return 0.0
    preds = np.array(predictions) > threshold
    trgs = np.array(targets)
    return float((preds == trgs).mean())


def run_epoch(
    model: torch.nn.Module,
    loader: Iterable,
    criterion: torch.nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    *,
    use_tqdm: bool = False,
    desc: str | None = None,
    threshold: float = 0.5,
) -> Tuple[float, float, List[float], List[float]]:
    """Run a full epoch for training or evaluation and return loss/accuracy details."""
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    iterator = tqdm(loader, desc=desc) if use_tqdm else loader
    running_loss = 0.0
    predictions: List[float] = []
    targets: List[float] = []

    grad_ctx = torch.enable_grad() if is_train else torch.no_grad()
    with grad_ctx:
        for batch in iterator:
            inputs, labels = batch
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs).squeeze()
            loss = criterion(outputs, labels)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            running_loss += loss.item()
            probs = torch.sigmoid(outputs).detach().cpu().numpy().tolist()
            predictions.extend(probs)
            targets.extend(labels.detach().cpu().numpy().tolist())

            if use_tqdm:
                iterator.set_postfix({'loss': loss.item()})

    num_batches = len(loader) if hasattr(loader, '__len__') else None
    denom = num_batches if num_batches and num_batches > 0 else max(len(predictions), 1)
    avg_loss = running_loss / denom
    accuracy = compute_accuracy(predictions, targets, threshold)
    return avg_loss, accuracy, predictions, targets

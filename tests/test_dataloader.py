"""Tests for the preprocessing dataloader utilities."""

from pathlib import Path
import sys

import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocess.data_handler import get_dataloaders


def _create_dummy_dataset(root: Path, *, num_samples: int = 4) -> None:
    (root / 'real').mkdir(parents=True, exist_ok=True)
    (root / 'fake').mkdir(parents=True, exist_ok=True)
    for idx in range(num_samples):
        for split, color in [('real', (255, 0, 0)), ('fake', (0, 255, 0))]:
            image = Image.new('RGB', (32, 32), color=color)
            image.save(root / split / f'{split}_{idx}.png')


def test_get_dataloaders_returns_pytorch_dataloaders(tmp_path):
    dataset_root = tmp_path / 'processed'
    _create_dummy_dataset(dataset_root, num_samples=4)

    train_loader, val_loader, test_loader = get_dataloaders(
        str(dataset_root),
        image_size=32,
        batch_size=2,
    )

    for loader in (train_loader, val_loader, test_loader):
        batch = next(iter(loader))
        images, labels = batch
        assert isinstance(images, torch.Tensor)
        assert isinstance(labels, torch.Tensor)
        assert images.shape[1:] == (3, 32, 32)

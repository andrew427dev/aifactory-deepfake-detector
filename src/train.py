"""Unified training entrypoint for the deepfake detector models."""
from __future__ import annotations

import argparse
from types import SimpleNamespace
from typing import Callable, Dict

import torch

from src.models import (
    cnn_transformer,
    cross_attention,
    efficientnet,
    swin,
    xception,
)

MODEL_REGISTRY: Dict[str, Callable[[SimpleNamespace], None]] = {
    'xception': xception.run_training,
    'swin': swin.run_training,
    'efficientnet': efficientnet.run_training,
    'cnn_transformer': cnn_transformer.run_training,
    'cross_attention': cross_attention.run_training,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Train a deepfake detector model from the src package.'
    )
    parser.add_argument(
        '--model',
        required=True,
        choices=MODEL_REGISTRY.keys(),
        help='Model architecture to train.',
    )
    parser.add_argument(
        '--data-dir',
        default='data/processed',
        help='Path to the dataset root containing real/ and fake/ folders.',
    )
    parser.add_argument('--image-size', type=int, help='Override input image size.')
    parser.add_argument('--batch-size', type=int, help='Override batch size.')
    parser.add_argument('--epochs', type=int, help='Override epoch count.')
    parser.add_argument(
        '--experiment-name',
        help='Optional MLflow experiment name override.',
    )
    parser.add_argument(
        '--tracking-uri',
        help='Optional MLflow tracking URI (defaults to file:./mlruns).',
    )
    parser.add_argument('--seed', type=int, help='Random seed override.')
    parser.add_argument(
        '--device',
        choices=['cpu', 'cuda'],
        help='Force training device. Defaults to auto-detect.',
    )
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        data_dir=args.data_dir,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        experiment_name=args.experiment_name,
        tracking_uri=args.tracking_uri,
        seed=args.seed,
        device=torch.device(args.device) if args.device else None,
    )


def main() -> None:
    args = parse_args()
    config = build_config(args)

    trainer = MODEL_REGISTRY[args.model]
    trainer(config)


if __name__ == '__main__':
    main()

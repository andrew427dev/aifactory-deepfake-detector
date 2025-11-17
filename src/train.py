"""Unified training entrypoint for the deepfake detector models."""
from __future__ import annotations

import argparse
import os
from types import SimpleNamespace
from typing import Callable, Dict

from src.models import (
    cnn_transformer,
    cross_attention,
    efficientnet,
    swin,
    xception,
)
from src.utils import PROJECT_ROOT, get_device, load_yaml_config, to_absolute_path

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
    parser.add_argument('--config', default='config.yaml', help='Path to the YAML config file.')
    parser.add_argument(
        '--model',
        choices=MODEL_REGISTRY.keys(),
        help='Model architecture to train (overrides config).',
    )
    parser.add_argument(
        '--data-dir',
        help='(Deprecated) dataset root containing real/ and fake/ folders.',
    )
    parser.add_argument('--train-data-dir', help='Training directory override.')
    parser.add_argument('--val-data-dir', help='Validation directory override.')
    parser.add_argument('--checkpoint-dir', help='Checkpoint directory override.')
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
        choices=['cpu', 'cuda', 'mps'],
        help='Force training device. Defaults to auto-detect.',
    )
    return parser.parse_args()


def _resolve_directory(*segments: str) -> str:
    base_path = os.fspath(PROJECT_ROOT)
    return os.path.join(base_path, *segments)


def build_config(args: argparse.Namespace) -> SimpleNamespace:
    config_data = load_yaml_config(args.config)

    model_name = args.model or config_data.get('model')
    if model_name not in MODEL_REGISTRY:
        raise ValueError('A valid model name must be specified in args or config.yaml')

    default_train_dir = _resolve_directory('data', 'processed')
    train_dir_setting = (
        args.train_data_dir
        or args.data_dir
        or config_data.get('train_data_dir')
        or config_data.get('data_dir')
    )
    train_dir = to_absolute_path(train_dir_setting) or default_train_dir
    val_dir_setting = args.val_data_dir or config_data.get('val_data_dir')
    val_dir = to_absolute_path(val_dir_setting)

    checkpoint_dir_setting = args.checkpoint_dir or config_data.get('checkpoint_dir') or 'models'
    checkpoint_dir = to_absolute_path(checkpoint_dir_setting) or _resolve_directory('models')

    image_size = args.image_size or config_data.get('input_size') or 224
    batch_size = args.batch_size or config_data.get('batch_size') or 32
    num_epochs = args.epochs or config_data.get('epochs') or 1

    return SimpleNamespace(
        model=model_name,
        data_dir=train_dir,
        train_data_dir=train_dir,
        val_data_dir=val_dir,
        image_size=image_size,
        batch_size=batch_size,
        num_epochs=num_epochs,
        experiment_name=args.experiment_name or config_data.get('experiment_name'),
        tracking_uri=args.tracking_uri or config_data.get('tracking_uri'),
        seed=args.seed or config_data.get('seed'),
        device=get_device(args.device),
        checkpoint_dir=checkpoint_dir,
        checkpoint_path=os.path.join(checkpoint_dir, f'best_{model_name}.pt'),
        learning_rate=config_data.get('learning_rate'),
    )


def main() -> None:
    args = parse_args()
    config = build_config(args)

    trainer = MODEL_REGISTRY[config.model]
    trainer(config)


if __name__ == '__main__':
    main()

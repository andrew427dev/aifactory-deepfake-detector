"""Run inference on a trained deepfake detector model using YAML config settings."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader
from PIL import Image

from src.models.cnn_transformer import DeepfakeCNNTransformer
from src.models.cross_attention import DeepfakeCrossAttention
from src.models.efficientnet import DeepfakeEfficientNet
from src.models.swin import DeepfakeSwin
from src.models.xception import DeepfakeXception
from src.preprocess.data_handler import DeepfakeDataset, get_transforms
from src.utils import (
    PROJECT_ROOT,
    compute_accuracy,
    get_device,
    load_model,
    load_yaml_config,
    set_seed,
    to_absolute_path,
)

MODEL_FACTORIES = {
    'xception': DeepfakeXception,
    'swin': DeepfakeSwin,
    'efficientnet': DeepfakeEfficientNet,
    'cnn_transformer': DeepfakeCNNTransformer,
    'cross_attention': DeepfakeCrossAttention,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Run inference for a trained deepfake model.')
    parser.add_argument('--config', default='config.yaml', help='Path to the YAML config file.')
    parser.add_argument('--model', choices=MODEL_FACTORIES.keys(), help='Model to evaluate.')
    parser.add_argument('--checkpoint', help='Specific checkpoint path to load.')
    parser.add_argument('--checkpoint-dir', help='Directory containing checkpoints.')
    parser.add_argument('--data-dir', help='Directory to use for inference samples.')
    parser.add_argument('--batch-size', type=int, help='Override batch size for inference.')
    parser.add_argument('--device', choices=['cpu', 'cuda', 'mps'], help='Device override.')
    parser.add_argument('--seed', type=int, help='Random seed override.')
    return parser.parse_args()


def _default_models_dir() -> str:
    return os.path.join(os.fspath(PROJECT_ROOT), 'models')


def build_settings(args: argparse.Namespace) -> SimpleNamespace:
    config_data = load_yaml_config(args.config)

    model_name = args.model or config_data.get('model')
    if model_name not in MODEL_FACTORIES:
        raise ValueError('A valid model must be provided via --model or config.yaml')

    checkpoint_dir_setting = args.checkpoint_dir or config_data.get('checkpoint_dir') or 'models'
    checkpoint_dir = to_absolute_path(checkpoint_dir_setting) or _default_models_dir()
    checkpoint_path = args.checkpoint or os.path.join(checkpoint_dir, f'best_{model_name}.pt')

    data_dir_setting = args.data_dir or config_data.get('val_data_dir') or config_data.get('train_data_dir')
    data_dir = to_absolute_path(data_dir_setting)
    if data_dir is None:
        raise ValueError('No inference data directory configured.')

    batch_size = args.batch_size or config_data.get('batch_size') or 32
    image_size = config_data.get('input_size') or 224
    seed = args.seed or config_data.get('seed')

    return SimpleNamespace(
        model=model_name,
        checkpoint_path=checkpoint_path,
        data_dir=data_dir,
        batch_size=batch_size,
        image_size=image_size,
        seed=seed,
        device=get_device(args.device),
    )


def run_inference(settings: SimpleNamespace) -> None:
    set_seed(settings.seed)

    model_cls = MODEL_FACTORIES[settings.model]
    model = model_cls()
    model = load_model(model, settings.checkpoint_path, settings.device)
    model.eval()

    transform = get_transforms(settings.image_size)
    dataset = DeepfakeDataset(settings.data_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=settings.batch_size, num_workers=4)

    predictions = []
    targets = []
    criterion = torch.nn.BCEWithLogitsLoss()
    total_loss = 0.0

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(settings.device)
            labels = labels.to(settings.device)
            logits = model(images).squeeze()
            loss = criterion(logits, labels)
            total_loss += loss.item()
            probs = torch.sigmoid(logits).detach().cpu().numpy().tolist()
            predictions.extend(probs)
            targets.extend(labels.detach().cpu().numpy().tolist())

    avg_loss = total_loss / max(len(dataloader), 1)
    accuracy = compute_accuracy(predictions, targets)
    print(f'Inference complete | Loss: {avg_loss:.4f} | Accuracy: {accuracy:.4f}')


def _find_sample_image(data_dir: str) -> Path | None:
    base = Path(data_dir)
    search_dirs = [base]
    for sub in ('real', 'fake'):
        search_dirs.append(base / sub)
    extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp')
    for directory in search_dirs:
        if not directory.is_dir():
            continue
        for ext in extensions:
            for candidate in sorted(directory.glob(f'*{ext}')):
                return candidate
    return None


def _simulate_inference_preview(settings: SimpleNamespace) -> None:
    image_path = _find_sample_image(settings.data_dir)
    transform = get_transforms(settings.image_size)
    if image_path is None:
        print(
            f'No images found in {settings.data_dir}. Using a random tensor for inference preview.'
        )
        sample = torch.randn(1, 3, settings.image_size, settings.image_size)
    else:
        with Image.open(image_path).convert('RGB') as image:
            sample = transform(image).unsqueeze(0)

    device = settings.device
    sample = sample.to(device)
    dummy_model = torch.nn.Sequential(
        torch.nn.Flatten(),
        torch.nn.Linear(sample.shape[1] * sample.shape[2] * sample.shape[3], 1),
        torch.nn.Sigmoid(),
    ).to(device)
    with torch.no_grad():
        prediction = dummy_model(sample)
    print(f'Inference preview device: {device}')
    print(f'Inference preview input shape: {tuple(sample.shape)}')
    print(f'Dummy prediction: {prediction.detach().cpu().numpy()}')


def main() -> None:
    args = parse_args()
    settings = build_settings(args)
    run_inference(settings)


if __name__ == '__main__':
    cli_args = parse_args()
    cli_settings = build_settings(cli_args)
    print(f'Loaded inference settings: {cli_settings}')
    _simulate_inference_preview(cli_settings)
    run_inference(cli_settings)

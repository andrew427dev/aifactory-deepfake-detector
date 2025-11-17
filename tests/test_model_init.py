"""Basic smoke tests to ensure models initialize and perform a forward pass."""

from pathlib import Path
import sys

import torch
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.xception import DeepfakeXception
from src.models.swin import DeepfakeSwin


@pytest.mark.parametrize(
    'model_cls,input_shape',
    [
        (DeepfakeXception, (2, 3, 299, 299)),
        (DeepfakeSwin, (2, 3, 224, 224)),
    ],
)
def test_model_initialization_and_forward(model_cls, input_shape):
    model = model_cls(pretrained=False)
    dummy = torch.randn(*input_shape)
    output = model(dummy)
    assert output.shape[0] == input_shape[0]
    assert output.shape[-1] == 1

from __future__ import annotations

import torch

from fl.parameters import get_parameters_as_numpy, load_parameters_into_model
from fl.factories import build_model_from_config
from segmentation.config import SegmentationConfig


def test_parameter_roundtrip_preserves_model_outputs() -> None:
    torch.manual_seed(42)
    config = SegmentationConfig()
    config.model.pretrained = False
    model = build_model_from_config(config)
    model.eval()
    x = torch.rand(1, 1, 256, 256)

    with torch.no_grad():
        before = model(x)
        if isinstance(before, tuple):
            before = before[0]

    params = get_parameters_as_numpy(model)
    load_parameters_into_model(model, params)

    with torch.no_grad():
        after = model(x)
        if isinstance(after, tuple):
            after = after[0]

    torch.testing.assert_close(before, after, rtol=0.0, atol=0.0)


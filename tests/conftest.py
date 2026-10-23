from importlib.resources import files

import numpy as np
import pytest

from datafarm_pyroki import load_encoder


@pytest.fixture(scope="session")
def encoder():
    return load_encoder()


@pytest.fixture(scope="session")
def torch_encoder():
    import torch
    from reference.torch_encoder import TrajectoryEncoder

    torch.set_num_threads(1)
    model = TrajectoryEncoder(28, 16, 113).eval()
    asset = files("datafarm_pyroki").joinpath("assets/droid_encoder.npz")
    with asset.open("rb") as stream, np.load(stream, allow_pickle=False) as blob:
        model.load_state_dict(
            {
                key.removeprefix("state_dict/"): torch.tensor(blob[key])
                for key in blob.files
                if key.startswith("state_dict/")
            }
        )
    for param in model.parameters():
        param.requires_grad_(False)
    return model


@pytest.fixture
def motion():
    t = np.linspace(0, 1, 32, dtype=np.float32)[:, None]
    base = np.array([0.1, -0.5, 0.2, -2.1, 0.1, 1.6, 0.8], dtype=np.float32)
    return base + 0.1 * np.sin(t * np.arange(1, 8, dtype=np.float32)[None])

import os
from pathlib import Path

import numpy as np
import pytest

from datafarm_pyroki import load_encoder
from datafarm_pyroki.assets import resolve_model_asset


def _checkpoint(filename):
    # CI without private Hub access may supply independently converted parity
    # fixtures. Production loaders never consult this test-only environment variable.
    directory = os.environ.get("DATAFARM_TEST_ASSET_DIR")
    return Path(directory) / filename if directory else resolve_model_asset(filename)


@pytest.fixture(scope="session")
def encoder_checkpoint():
    return _checkpoint("droid_encoder.npz")


@pytest.fixture(scope="session")
def joint_prior_checkpoint():
    return _checkpoint("droid_joint_prior.npz")


@pytest.fixture(scope="session")
def encoder(encoder_checkpoint):
    return load_encoder(encoder_checkpoint)


@pytest.fixture(scope="session")
def torch_encoder(encoder_checkpoint):
    import torch
    from reference.torch_encoder import TrajectoryEncoder

    torch.set_num_threads(1)
    model = TrajectoryEncoder(28, 16, 113).eval()
    with np.load(encoder_checkpoint, allow_pickle=False) as blob:
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

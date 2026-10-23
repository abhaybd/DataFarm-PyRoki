"""Resolve external model files without embedding checkpoint binaries in the package."""

from __future__ import annotations

import os
from contextlib import contextmanager
from importlib.resources import files
from pathlib import Path


@contextmanager
def open_model_asset(filename: str, path: str | Path | None = None):
    """An explicit file overrides DATAFARM_ASSET_DIR, then the local source asset folder.

    Source-tree assets may exist locally but are gitignored and absent from wheels.
    Installed packages normally use DATAFARM_ASSET_DIR or explicit model paths.
    """
    directory = os.environ.get("DATAFARM_ASSET_DIR")
    if path is not None:
        resource = Path(path).expanduser()
    elif directory:
        resource = Path(directory).expanduser() / filename
    else:
        resource = files("datafarm_pyroki").joinpath(f"assets/{filename}")
    try:
        stream = resource.open("rb")
    except FileNotFoundError as error:
        raise FileNotFoundError(
            f"Model asset {filename!r} was not found at {resource}. Checkpoints are distributed "
            "separately from Git/the wheel. Pass an explicit model path or set DATAFARM_ASSET_DIR "
            "to your downloaded weights folder. In a source checkout, use scripts/fetch_assets.py "
            "with your artifact URLs or the scripts/convert_*.py reproduction tools."
        ) from error
    with stream:
        yield stream

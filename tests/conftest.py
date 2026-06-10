"""
Shared fixtures and utilities for unit and integration tests.
"""
import random
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch


@pytest.fixture
def temp_dir():
    """Temporary directory cleaned up after the test."""
    temp_path = tempfile.mkdtemp()
    yield Path(temp_path)
    shutil.rmtree(temp_path)


@pytest.fixture
def sample_image_tensor():
    """Single image tensor (1, 3, 64, 64)."""
    return torch.randn(1, 3, 64, 64)


@pytest.fixture
def sample_batch():
    """Mini-batch of 4 images with random labels (4 classes)."""
    return {
        "images": torch.randn(4, 3, 64, 64),
        "labels": torch.randint(0, 4, (4,)),
    }


@pytest.fixture
def device():
    """CPU or CUDA device string."""
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture
def reset_seed():
    """Reset all random seeds to 0 after the test."""
    yield
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)


@pytest.fixture(scope="session")
def synthetic_dataset(tmp_path_factory):
    """
    Session-scoped minimal ImageFolder-compatible dataset.

    Structure created under a temp directory:
        <root>/
        ├── classes.txt           (apple, banana)
        ├── fold0/
        │   ├── train/apple/  (4 tiny JPEG images)
        │   ├── train/banana/
        │   ├── val/apple/
        │   └── val/banana/
        ├── train/apple/          (for fold=None mode="train")
        ├── train/banana/
        ├── val/apple/
        ├── val/banana/
        └── test/apple/           (for fold=None mode="test")
            test/banana/

    Images are 16×16 RGB JPEGs created with Pillow (no real photo content).
    """
    from PIL import Image  # pillow is in the dev group

    root = tmp_path_factory.mktemp("synthetic_data")
    classes = ["apple", "banana"]
    n_images = 4

    splits = ["fold0/train", "fold0/val", "train", "val", "test"]
    for split in splits:
        for cls in classes:
            d = root / split / cls
            d.mkdir(parents=True)
            for i in range(n_images):
                color = (i * 40 + 80, i * 30 + 60, 120)
                img = Image.new("RGB", (16, 16), color=color)
                img.save(d / f"img_{i:03d}.jpg")

    (root / "classes.txt").write_text("\n".join(classes) + "\n")
    return root

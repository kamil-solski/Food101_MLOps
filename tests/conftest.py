"""
Shared fixtures and utilities for unit tests.
"""
import pytest
import torch
import numpy as np
import tempfile
import shutil
from pathlib import Path
import os


@pytest.fixture
def temp_dir():
    """Create a temporary directory for testing."""
    temp_path = tempfile.mkdtemp()
    yield Path(temp_path)
    shutil.rmtree(temp_path)


@pytest.fixture
def sample_image_tensor():
    """Create a sample image tensor for testing."""
    return torch.randn(1, 3, 64, 64)


@pytest.fixture
def sample_batch():
    """Create a sample batch of data."""
    return {
        'images': torch.randn(4, 3, 64, 64),
        'labels': torch.randint(0, 4, (4,))
    }


@pytest.fixture
def device():
    """Get the device (CPU or CUDA) for testing."""
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture
def reset_seed():
    """Fixture to reset random seeds after each test."""
    yield
    # Reset to a known state
    import random
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

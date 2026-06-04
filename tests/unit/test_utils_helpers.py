"""
Unit tests for src.utils.helpers module.
"""
import pytest
import random
import numpy as np
import torch
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils.helpers import set_seed


class TestSetSeed:
    """Test suite for set_seed function."""

    def test_set_seed(self):
        """Verify seed sets correctly."""
        seed = 42
        set_seed(seed)
        
        # Verify Python random seed
        value1 = random.random()
        random.seed(seed)
        value2 = random.random()
        assert value1 == value2
        
        # Verify NumPy random seed
        value1 = np.random.random()
        np.random.seed(seed)
        value2 = np.random.random()
        assert value1 == value2
        
        # Verify PyTorch random seed
        value1 = torch.rand(1).item()
        torch.manual_seed(seed)
        value2 = torch.rand(1).item()
        assert value1 == value2

    def test_set_seed_reproducibility(self):
        """Same seed produces same results."""
        seed = 123
        
        # First run
        set_seed(seed)
        python_val1 = random.random()
        numpy_val1 = np.random.random()
        torch_val1 = torch.rand(1).item()
        
        # Second run with same seed
        set_seed(seed)
        python_val2 = random.random()
        numpy_val2 = np.random.random()
        torch_val2 = torch.rand(1).item()
        
        assert python_val1 == python_val2
        assert numpy_val1 == numpy_val2
        assert torch_val1 == torch_val2

    def test_set_seed_different_seeds(self):
        """Different seeds produce different results."""
        set_seed(42)
        python_val1 = random.random()
        numpy_val1 = np.random.random()
        torch_val1 = torch.rand(1).item()
        
        set_seed(999)
        python_val2 = random.random()
        numpy_val2 = np.random.random()
        torch_val2 = torch.rand(1).item()
        
        assert python_val1 != python_val2
        assert numpy_val1 != numpy_val2
        assert torch_val1 != torch_val2

    def test_set_seed_all_generators(self):
        """Python, NumPy, PyTorch all seeded."""
        seed = 42
        set_seed(seed)
        
        # Generate values from all generators
        python_val = random.random()
        numpy_val = np.random.random()
        torch_val = torch.rand(1).item()
        
        # Reset and verify they're reproducible
        set_seed(seed)
        assert python_val == random.random()
        assert numpy_val == np.random.random()
        assert torch_val == torch.rand(1).item()

    def test_set_seed_cuda_deterministic(self):
        """CUDA deterministic mode enabled."""
        seed = 42
        set_seed(seed)
        
        # Check deterministic settings
        assert torch.backends.cudnn.deterministic == True
        assert torch.backends.cudnn.benchmark == False
        
        # Check environment variable
        assert os.environ.get('PYTHONHASHSEED') == str(seed)
        
        # If CUDA is available, verify CUDA seeds are set
        if torch.cuda.is_available():
            # Generate CUDA tensor and verify reproducibility
            set_seed(seed)
            val1 = torch.rand(1, device='cuda').item()
            set_seed(seed)
            val2 = torch.rand(1, device='cuda').item()
            assert val1 == val2

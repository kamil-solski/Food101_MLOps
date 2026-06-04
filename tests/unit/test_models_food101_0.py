"""
Unit tests for src.models.food101_0 module.
"""
import pytest
import torch
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.food101_0 import Food101_0
from src.models.food101 import Food101


class TestFood101_0:
    """Test suite for Food101_0 model."""

    def test_food101_0_initialization(self):
        """Model initialization."""
        model = Food101_0(input_shape=3, hidden_units=16, output_shape=10)
        
        assert model is not None
        assert hasattr(model, 'conv_block_1')
        assert hasattr(model, 'conv_block_2')
        assert hasattr(model, 'classifier')
        # Food101_0 should NOT have conv_block_3
        assert not hasattr(model, 'conv_block_3')

    def test_food101_0_forward_pass(self):
        """Mechanical forward pass test."""
        model = Food101_0(input_shape=3, hidden_units=8, output_shape=4)
        model.eval()
        
        x = torch.randn(2, 3, 64, 64)
        
        with torch.no_grad():
            output = model(x)
        
        assert output is not None
        assert isinstance(output, torch.Tensor)
        assert len(output.shape) == 2  # (batch_size, num_classes)

    def test_food101_0_output_shape(self):
        """Output shape validation."""
        batch_size = 4
        input_channels = 3
        hidden_units = 16
        num_classes = 10
        
        model = Food101_0(
            input_shape=input_channels,
            hidden_units=hidden_units,
            output_shape=num_classes
        )
        model.eval()
        
        x = torch.randn(batch_size, input_channels, 64, 64)
        
        with torch.no_grad():
            output = model(x)
        
        assert output.shape == (batch_size, num_classes)

    def test_food101_0_vs_food101_difference(self):
        """Architecture differences."""
        # Both models with same parameters
        model_0 = Food101_0(input_shape=3, hidden_units=8, output_shape=4)
        model_full = Food101(input_shape=3, hidden_units=8, output_shape=4)
        
        # Food101_0 has 2 conv blocks, Food101 has 3
        assert hasattr(model_0, 'conv_block_1')
        assert hasattr(model_0, 'conv_block_2')
        assert not hasattr(model_0, 'conv_block_3')
        
        assert hasattr(model_full, 'conv_block_1')
        assert hasattr(model_full, 'conv_block_2')
        assert hasattr(model_full, 'conv_block_3')
        
        # Count parameters - Food101_0 has fewer conv blocks, but different linear layer size
        # Let's just verify they have different parameter counts and both have parameters
        params_0 = sum(p.numel() for p in model_0.parameters())
        params_full = sum(p.numel() for p in model_full.parameters())
        
        assert params_0 > 0
        assert params_full > 0
        # They should have different parameter counts due to architecture differences
        assert params_0 != params_full
        
        # Test with same input
        x = torch.randn(2, 3, 64, 64)
        model_0.eval()
        model_full.eval()
        
        with torch.no_grad():
            out_0 = model_0(x)
            out_full = model_full(x)
        
        # Both should produce same shape output
        assert out_0.shape == out_full.shape == (2, 4)

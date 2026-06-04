"""
Unit tests for src.models.food101 module.
"""
import pytest
import torch
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.food101 import Food101


class TestFood101:
    """Test suite for Food101 model."""

    def test_food101_initialization(self):
        """Model initialization."""
        model = Food101(input_shape=3, hidden_units=16, output_shape=10)
        
        assert model is not None
        assert hasattr(model, 'conv_block_1')
        assert hasattr(model, 'conv_block_2')
        assert hasattr(model, 'conv_block_3')
        assert hasattr(model, 'classifier')

    def test_food101_forward_pass(self):
        """Forward pass shape validation."""
        model = Food101(input_shape=3, hidden_units=8, output_shape=4)
        model.eval()
        
        # Input: (batch_size, channels, height, width)
        x = torch.randn(2, 3, 64, 64)
        
        with torch.no_grad():
            output = model(x)
        
        assert output is not None
        assert isinstance(output, torch.Tensor)

    def test_food101_output_shape(self):
        """Correct output dimensions."""
        batch_size = 4
        input_channels = 3
        hidden_units = 16
        num_classes = 10
        
        model = Food101(
            input_shape=input_channels,
            hidden_units=hidden_units,
            output_shape=num_classes
        )
        model.eval()
        
        x = torch.randn(batch_size, input_channels, 64, 64)
        
        with torch.no_grad():
            output = model(x)
        
        assert output.shape == (batch_size, num_classes)

    def test_food101_gradient_flow(self):
        """Gradients computed correctly."""
        model = Food101(input_shape=3, hidden_units=8, output_shape=4)
        model.train()
        
        x = torch.randn(2, 3, 64, 64, requires_grad=True)
        target = torch.randint(0, 4, (2,))
        
        output = model(x)
        loss = torch.nn.functional.cross_entropy(output, target)
        loss.backward()
        
        # Check that gradients exist
        for param in model.parameters():
            assert param.grad is not None
            assert not torch.isnan(param.grad).any()

    def test_food101_device_placement(self, device):
        """CPU/GPU device handling."""
        model = Food101(input_shape=3, hidden_units=8, output_shape=4)
        model = model.to(device)
        
        x = torch.randn(2, 3, 64, 64, device=device)
        
        with torch.no_grad():
            output = model(x)
        
        assert output.device.type == device

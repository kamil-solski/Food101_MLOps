"""
Unit tests for inference_api.inference.predictor module.
"""
import pytest
import numpy as np
import io
from PIL import Image
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from inference_api.inference.predictor import _preprocess, _softmax


class TestPreprocess:
    """Test suite for _preprocess function."""

    @pytest.fixture
    def sample_image_bytes(self):
        """Create sample image bytes."""
        img = Image.new('RGB', (128, 128), color='red')
        buf = io.BytesIO()
        img.save(buf, format='PNG')
        return buf.getvalue()

    def test_preprocess_image(self, sample_image_bytes):
        """Image preprocessing."""
        image_size = 64
        cfg = {"preprocess": {"scale_to_unit": True}}
        
        result = _preprocess(sample_image_bytes, image_size, cfg)
        
        assert result.shape == (1, 3, image_size, image_size)
        assert result.dtype == np.float32

    def test_preprocess_image_size(self, sample_image_bytes):
        """Image resizing."""
        cfg = {"preprocess": {"scale_to_unit": True}}
        
        result = _preprocess(sample_image_bytes, 64, cfg)
        assert result.shape[2] == 64
        assert result.shape[3] == 64
        
        result = _preprocess(sample_image_bytes, 128, cfg)
        assert result.shape[2] == 128
        assert result.shape[3] == 128

    def test_preprocess_normalization(self, sample_image_bytes):
        """Normalization."""
        cfg = {
            "preprocess": {
                "scale_to_unit": True,
                "normalize": True,
                "mean": [0.5, 0.5, 0.5],
                "std": [0.5, 0.5, 0.5]
            }
        }
        
        result = _preprocess(sample_image_bytes, 64, cfg)
        
        assert result.shape == (1, 3, 64, 64)
        # Values should be normalized (not in [0, 1] anymore)
        assert result.min() < 0 or result.max() > 1  # After normalization

    def test_preprocess_scale_to_unit(self, sample_image_bytes):
        """Unit scaling."""
        cfg = {"preprocess": {"scale_to_unit": True}}
        
        result = _preprocess(sample_image_bytes, 64, cfg)
        
        # Values should be in [0, 1] range
        assert result.min() >= 0
        assert result.max() <= 1.0


class TestSoftmax:
    """Test suite for _softmax function."""

    def test_softmax(self):
        """Softmax function."""
        logits = np.array([1.0, 2.0, 3.0])
        probs = _softmax(logits)
        
        assert len(probs) == 3
        assert np.allclose(probs.sum(), 1.0, atol=1e-6)
        assert np.all(probs >= 0)
        assert np.all(probs <= 1)
        # Highest logit should have highest probability
        assert probs[2] > probs[1] > probs[0]

    def test_softmax_edge_cases(self):
        """Edge cases (zeros, large values)."""
        # Test with zeros
        logits = np.array([0.0, 0.0, 0.0])
        probs = _softmax(logits)
        
        assert np.allclose(probs, [1/3, 1/3, 1/3], atol=1e-6)
        
        # Test with large values (numerical stability)
        logits = np.array([1000.0, 1001.0, 1002.0])
        probs = _softmax(logits)
        
        assert not np.isnan(probs).any()
        assert not np.isinf(probs).any()
        assert np.allclose(probs.sum(), 1.0, atol=1e-6)

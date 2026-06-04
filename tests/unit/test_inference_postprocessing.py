"""
Unit tests for inference_api.utils.postprocessing module.
"""
import pytest
import numpy as np
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from inference_api.utils.postprocessing import topk_indices, map_indices_to_labels


class TestTopkIndices:
    """Test suite for topk_indices function."""

    def test_topk_indices_basic(self):
        """Basic top-k selection."""
        probs = np.array([0.1, 0.5, 0.2, 0.15, 0.05])
        k = 3
        
        indices = topk_indices(probs, k)
        
        assert len(indices) == 3
        assert indices[0] == 1  # Highest probability at index 1
        assert 1 in indices
        assert 2 in indices  # Second highest

    def test_topk_indices_edge_cases(self):
        """k > array size, k=0, k negative."""
        probs = np.array([0.3, 0.5, 0.2])
        
        # k > array size should clamp to array size
        indices = topk_indices(probs, k=10)
        assert len(indices) == 3
        
        # k = 0 should default to 1
        indices = topk_indices(probs, k=0)
        assert len(indices) == 1
        
        # k negative should default to 1
        indices = topk_indices(probs, k=-5)
        assert len(indices) == 1

    def test_topk_indices_sorted_order(self):
        """Descending order."""
        probs = np.array([0.1, 0.5, 0.2, 0.3, 0.05])
        k = 3
        
        indices = topk_indices(probs, k)
        
        # Verify descending order
        probs_sorted = probs[indices]
        assert all(probs_sorted[i] >= probs_sorted[i+1] 
                  for i in range(len(probs_sorted)-1))


class TestMapIndicesToLabels:
    """Test suite for map_indices_to_labels function."""

    def test_map_indices_to_labels(self):
        """Label mapping."""
        probs = np.array([0.1, 0.5, 0.2, 0.15, 0.05])
        indices = np.array([1, 2, 0])
        labels = ["apple", "banana", "cherry", "date", "elderberry"]
        
        result = map_indices_to_labels(indices, probs, labels)
        
        assert len(result) == 3
        assert result[0]["class"] == "banana"
        assert result[0]["id"] == 1
        assert result[0]["prob"] == 0.5
        assert result[1]["class"] == "cherry"
        assert result[2]["class"] == "apple"

    def test_map_indices_to_labels_missing_labels(self):
        """Missing labels handling."""
        probs = np.array([0.1, 0.5, 0.2, 0.15, 0.05])
        indices = np.array([0, 3, 4])
        labels = ["apple", "banana"]  # Only 2 labels, but 5 probabilities
        
        result = map_indices_to_labels(indices, probs, labels)
        
        assert len(result) == 3
        assert result[0]["class"] == "apple"
        assert result[1]["class"] == "3"  # Falls back to index string
        assert result[2]["class"] == "4"  # Falls back to index string

    def test_map_indices_to_labels_empty(self):
        """Empty inputs."""
        probs = np.array([0.5, 0.3, 0.2])
        indices = np.array([])
        
        result = map_indices_to_labels(indices, probs)
        
        assert len(result) == 0
        
        # Test with None labels
        indices = np.array([0, 1])
        result = map_indices_to_labels(indices, probs, labels=None)
        
        assert len(result) == 2
        assert result[0]["class"] == "0"
        assert result[1]["class"] == "1"

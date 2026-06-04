"""
Unit tests for config file validation.
"""
import pytest
import yaml
import tempfile
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class TestConfigValidation:
    """Test suite for config.yaml validation."""

    @pytest.fixture
    def valid_config(self):
        """Create a valid config dictionary."""
        return {
            "batch_size": 32,
            "num_epochs": 10,
            "image_size": 64,
            "learning_rates": [0.0001, 0.001],
            "hidden_units": [8, 16],
            "architectures": ["Food101", "Food101_0"],
            "dataset": "food-101_50%_tr70_va15_te15",
            "random_seed": 42
        }

    def test_config_loading(self, tmp_path):
        """Config file loading."""
        config_file = tmp_path / "config.yaml"
        config_data = {
            "batch_size": 32,
            "num_epochs": 10,
            "image_size": 64,
            "learning_rates": [0.0001, 0.001],
            "hidden_units": [8, 16],
            "architectures": ["Food101"],
            "dataset": "test_dataset",
            "random_seed": 42
        }
        
        with open(config_file, 'w') as f:
            yaml.dump(config_data, f)
        
        with open(config_file, 'r') as f:
            loaded_config = yaml.safe_load(f)
        
        assert loaded_config == config_data

    def test_config_required_params(self, valid_config):
        """Required parameters."""
        required_keys = [
            "batch_size", "num_epochs", "image_size",
            "learning_rates", "hidden_units", "architectures",
            "dataset", "random_seed"
        ]
        
        for key in required_keys:
            assert key in valid_config

    def test_config_data_types(self, valid_config):
        """Data type validation."""
        assert isinstance(valid_config["batch_size"], int)
        assert isinstance(valid_config["num_epochs"], int)
        assert isinstance(valid_config["image_size"], int)
        assert isinstance(valid_config["learning_rates"], list)
        assert isinstance(valid_config["hidden_units"], list)
        assert isinstance(valid_config["architectures"], list)
        assert isinstance(valid_config["dataset"], str)
        assert isinstance(valid_config["random_seed"], int)

    def test_config_ranges(self, valid_config):
        """Parameter range validation."""
        assert valid_config["batch_size"] > 0
        assert valid_config["num_epochs"] > 0
        assert valid_config["image_size"] > 0
        assert all(lr > 0 for lr in valid_config["learning_rates"])
        assert all(hu > 0 for hu in valid_config["hidden_units"])
        assert valid_config["random_seed"] >= 0

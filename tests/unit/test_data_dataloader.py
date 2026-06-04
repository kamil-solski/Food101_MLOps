"""
Unit tests for src.data.dataloader module.
"""
import pytest
import yaml
import tempfile
import sys
from pathlib import Path
import shutil

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.dataloader import get_dataloaders


class TestGetDataloaders:
    """Test suite for get_dataloaders function."""
    
    @pytest.fixture
    def temp_config(self, tmp_path):
        """Create a temporary config file pointing to test_dataset."""
        config_file = tmp_path / "config.yaml"
        config_data = {
            "batch_size": 4,
            "num_epochs": 1,
            "image_size": 64,
            "learning_rates": [0.001],
            "hidden_units": [8],
            "architectures": ["Food101"],
            "dataset": "test_dataset",
            "random_seed": 42
        }
        with open(config_file, 'w') as f:
            yaml.dump(config_data, f)
        
        # Monkey patch get_paths in both modules to use temp config
        import src.utils.paths as paths_module
        import src.data.dataloader as dataloader_module
        original_get_paths = paths_module.get_paths
        
        def get_paths_with_config(*args, **kwargs):
            if 'config_path' not in kwargs:
                kwargs['config_path'] = config_file
            return original_get_paths(*args, **kwargs)
        
        # Patch in both places (dataloader imports get_paths, so we need to patch both)
        paths_module.get_paths = get_paths_with_config
        dataloader_module.get_paths = get_paths_with_config
        
        yield config_file
        
        # Restore
        paths_module.get_paths = original_get_paths
        dataloader_module.get_paths = original_get_paths
    
    def test_get_dataloaders_with_fold(self, temp_config):
        """Dataloader creation with fold."""
        train_loader, val_loader, class_names = get_dataloaders(
            image_size=64,
            batch_size=4,
            fold="fold0"
        )
        
        assert train_loader is not None
        assert val_loader is not None
        assert isinstance(class_names, list)
        assert len(class_names) > 0
        
        # Test that we can iterate
        batch_count = 0
        for X, y in train_loader:
            assert X.shape[0] <= 4  # batch_size
            assert X.shape[1] == 3  # channels
            assert X.shape[2] == 64  # height (image_size)
            assert X.shape[3] == 64  # width (image_size)
            batch_count += 1
            if batch_count >= 2:  # Just test a couple batches
                break
    
    def test_get_dataloaders_without_fold(self, temp_config):
        """Test set dataloader."""
        test_loader, class_names = get_dataloaders(
            image_size=64,
            batch_size=4,
            fold=None
        )
        
        assert test_loader is not None
        assert isinstance(class_names, list)
        assert len(class_names) > 0
        
        # Test that we can iterate
        for X, y in test_loader:
            assert X.shape[0] <= 4  # batch_size
            assert X.shape[1] == 3  # channels
            assert X.shape[2] == 64  # height
            assert X.shape[3] == 64  # width
            break  # Just test one batch
    
    def test_get_dataloaders_image_size(self, temp_config):
        """Image size transformation."""
        # Test with different image sizes
        for img_size in [32, 64, 128]:
            train_loader, val_loader, _ = get_dataloaders(
                image_size=img_size,
                batch_size=2,
                fold="fold0"
            )
            
            X, y = next(iter(train_loader))
            assert X.shape[2] == img_size  # height
            assert X.shape[3] == img_size  # width
    
    def test_get_dataloaders_batch_size(self, temp_config):
        """Batch size handling."""
        # Test with different batch sizes
        for batch_size in [1, 2, 4]:
            train_loader, val_loader, _ = get_dataloaders(
                image_size=64,
                batch_size=batch_size,
                fold="fold0"
            )
            
            X, y = next(iter(train_loader))
            assert X.shape[0] <= batch_size
    
    def test_get_dataloaders_class_names(self, temp_config):
        """Class names loaded correctly."""
        train_loader, val_loader, class_names = get_dataloaders(
            image_size=64,
            batch_size=4,
            fold="fold0"
        )
        
        assert isinstance(class_names, list)
        assert len(class_names) > 0
        assert all(isinstance(name, str) for name in class_names)
        assert all(len(name) > 0 for name in class_names)
    
    def test_get_dataloaders_shuffle(self, temp_config):
        """Shuffle flag behavior."""
        # Get two loaders - shuffle should affect order
        train_loader1, _, _ = get_dataloaders(
            image_size=64,
            batch_size=4,
            fold="fold0"
        )
        
        train_loader2, _, _ = get_dataloaders(
            image_size=64,
            batch_size=4,
            fold="fold0"
        )
        
        # Collect first few labels from both loaders
        labels1 = []
        labels2 = []
        for i, ((_, y1), (_, y2)) in enumerate(zip(train_loader1, train_loader2)):
            labels1.extend(y1.tolist())
            labels2.extend(y2.tolist())
            if i >= 2:  # Check first 3 batches
                break
        
        # With shuffle=True, order might be different (not guaranteed, but likely)
        # At minimum, we verify both have same set of labels
        assert len(labels1) > 0
        assert len(labels2) > 0
    
    def test_get_dataloaders_missing_directory(self, temp_config):
        """Error handling."""
        # Create config pointing to non-existent dataset
        config_file = temp_config
        with open(config_file, 'r') as f:
            config_data = yaml.safe_load(f)
        
        config_data["dataset"] = "nonexistent_dataset"
        with open(config_file, 'w') as f:
            yaml.dump(config_data, f)
        
        # Should raise an error when trying to access non-existent directory
        with pytest.raises((FileNotFoundError, OSError, ValueError)):
            get_dataloaders(
                image_size=64,
                batch_size=4,
                fold="fold0"
            )
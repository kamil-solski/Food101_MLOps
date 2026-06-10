"""
Unit tests for src.data.dataloader module.
"""
import pytest
import yaml
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.dataloader import get_dataloaders


class TestGetDataloaders:
    """Test suite for get_dataloaders function."""

    @pytest.fixture
    def patched_data_paths(self, tmp_path, synthetic_dataset):
        """
        Patch the paths module so get_dataloaders uses the synthetic dataset.
        Patches DATA_DIR, all OUTPUTS_* dirs, and injects the temp config_path.
        """
        import src.utils.paths as pm
        import src.data.dataloader as dm

        config_file = tmp_path / "config.yaml"
        with open(config_file, "w") as f:
            yaml.dump(
                {
                    "batch_size": 4,
                    "num_epochs": 1,
                    "image_size": 64,
                    "learning_rates": [0.001],
                    "hidden_units": [8],
                    "architectures": ["Food101"],
                    "dataset": synthetic_dataset.name,
                    "random_seed": 42,
                },
                f,
            )

        saved = {
            "DATA_DIR": pm.DATA_DIR,
            "OUTPUTS_DIR": pm.OUTPUTS_DIR,
            "CHECKPOINTS_DIR": pm.CHECKPOINTS_DIR,
            "METRICS_DIR": pm.METRICS_DIR,
            "LOGS_DIR": pm.LOGS_DIR,
            "FIGURES_DIR": pm.FIGURES_DIR,
            "PREDICTIONS_DIR": pm.PREDICTIONS_DIR,
            "get_paths": pm.get_paths,
        }

        pm.DATA_DIR = synthetic_dataset.parent
        pm.OUTPUTS_DIR = tmp_path / "outputs"
        pm.CHECKPOINTS_DIR = pm.OUTPUTS_DIR / "checkpoints"
        pm.METRICS_DIR = pm.OUTPUTS_DIR / "metrics"
        pm.LOGS_DIR = pm.OUTPUTS_DIR / "logs"
        pm.FIGURES_DIR = pm.OUTPUTS_DIR / "figures"
        pm.PREDICTIONS_DIR = pm.OUTPUTS_DIR / "predictions"

        original_get_paths = saved["get_paths"]

        def _get_paths(*args, **kwargs):
            kwargs.setdefault("config_path", config_file)
            return original_get_paths(*args, **kwargs)

        pm.get_paths = _get_paths
        dm.get_paths = _get_paths

        yield config_file

        pm.DATA_DIR = saved["DATA_DIR"]
        pm.OUTPUTS_DIR = saved["OUTPUTS_DIR"]
        pm.CHECKPOINTS_DIR = saved["CHECKPOINTS_DIR"]
        pm.METRICS_DIR = saved["METRICS_DIR"]
        pm.LOGS_DIR = saved["LOGS_DIR"]
        pm.FIGURES_DIR = saved["FIGURES_DIR"]
        pm.PREDICTIONS_DIR = saved["PREDICTIONS_DIR"]
        pm.get_paths = saved["get_paths"]
        dm.get_paths = saved["get_paths"]

    def test_get_dataloaders_with_fold(self, patched_data_paths):
        """Dataloader creation with fold."""
        train_loader, val_loader, class_names = get_dataloaders(
            image_size=32, batch_size=4, fold="fold0"
        )

        assert train_loader is not None
        assert val_loader is not None
        assert isinstance(class_names, list)
        assert len(class_names) > 0

        X, y = next(iter(train_loader))
        assert X.shape[0] <= 4
        assert X.shape[1] == 3
        assert X.shape[2] == 32
        assert X.shape[3] == 32

    def test_get_dataloaders_without_fold(self, patched_data_paths):
        """Test-set dataloader (fold=None, mode='test')."""
        test_loader, class_names = get_dataloaders(
            image_size=32, batch_size=4, fold=None, mode="test"
        )

        assert test_loader is not None
        assert isinstance(class_names, list)
        assert len(class_names) > 0

        X, y = next(iter(test_loader))
        assert X.shape[1] == 3
        assert X.shape[2] == 32
        assert X.shape[3] == 32

    def test_get_dataloaders_image_size(self, patched_data_paths):
        """Image size transformation."""
        for img_size in [16, 32]:
            train_loader, _, _ = get_dataloaders(
                image_size=img_size, batch_size=2, fold="fold0"
            )
            X, _ = next(iter(train_loader))
            assert X.shape[2] == img_size
            assert X.shape[3] == img_size

    def test_get_dataloaders_batch_size(self, patched_data_paths):
        """Batch size handling."""
        for batch_size in [1, 2, 4]:
            train_loader, _, _ = get_dataloaders(
                image_size=32, batch_size=batch_size, fold="fold0"
            )
            X, _ = next(iter(train_loader))
            assert X.shape[0] <= batch_size

    def test_get_dataloaders_class_names(self, patched_data_paths):
        """Class names loaded correctly."""
        _, _, class_names = get_dataloaders(
            image_size=32, batch_size=4, fold="fold0"
        )

        assert isinstance(class_names, list)
        assert len(class_names) > 0
        assert all(isinstance(name, str) for name in class_names)
        assert all(len(name) > 0 for name in class_names)

    def test_get_dataloaders_pixel_range(self, patched_data_paths):
        """ToTensor scales pixels to [0, 1]."""
        train_loader, _, _ = get_dataloaders(
            image_size=32, batch_size=4, fold="fold0"
        )
        X, _ = next(iter(train_loader))
        assert X.min() >= 0.0
        assert X.max() <= 1.0

    def test_get_dataloaders_missing_directory(self, patched_data_paths):
        """Error when dataset directory does not exist."""
        with open(patched_data_paths, "r") as f:
            config_data = yaml.safe_load(f)

        config_data["dataset"] = "nonexistent_dataset_xyz"
        with open(patched_data_paths, "w") as f:
            yaml.dump(config_data, f)

        with pytest.raises((FileNotFoundError, OSError, ValueError)):
            get_dataloaders(image_size=32, batch_size=4, fold="fold0")

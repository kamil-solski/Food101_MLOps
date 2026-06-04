"""
Unit tests for src.utils.paths module.
"""
import pytest
import yaml
import tempfile
import shutil
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils.paths import get_paths, clean_outputs_dir, OUTPUTS_DIR


class TestGetPaths:
    """Test suite for get_paths function."""

    @pytest.fixture
    def temp_project(self, tmp_path):
        """Create a temporary project structure."""
        # Create project structure
        project_root = tmp_path / "test_project"
        project_root.mkdir()
        
        # Create src directory with config
        src_dir = project_root / "src"
        src_dir.mkdir()
        
        config_file = src_dir / "config.yaml"
        config_data = {
            "dataset": "test_dataset",
            "image_size": 64,
            "batch_size": 32
        }
        with open(config_file, 'w') as f:
            yaml.dump(config_data, f)
        
        # Create Data directory
        data_dir = project_root / "Data"
        data_dir.mkdir()
        dataset_dir = data_dir / "test_dataset"
        dataset_dir.mkdir()
        
        # Create test directory
        test_dir = dataset_dir / "test"
        test_dir.mkdir()
        
        # Create classes.txt
        classes_file = dataset_dir / "classes.txt"
        classes_file.write_text("class1\nclass2\nclass3\n")
        
        # Monkey patch PROJECT_ROOT for this test
        import src.utils.paths as paths_module
        original_root = paths_module.PROJECT_ROOT
        paths_module.PROJECT_ROOT = project_root
        paths_module.DATA_DIR = project_root / "Data"
        paths_module.OUTPUTS_DIR = project_root / "outputs"
        paths_module.CHECKPOINTS_DIR = paths_module.OUTPUTS_DIR / "checkpoints"
        paths_module.METRICS_DIR = paths_module.OUTPUTS_DIR / "metrics"
        paths_module.LOGS_DIR = paths_module.OUTPUTS_DIR / "logs"
        paths_module.FIGURES_DIR = paths_module.OUTPUTS_DIR / "figures"
        paths_module.PREDICTIONS_DIR = paths_module.OUTPUTS_DIR / "predictions"
        
        yield project_root, config_file
        
        # Restore original
        paths_module.PROJECT_ROOT = original_root
        paths_module.DATA_DIR = original_root / "Data"
        paths_module.OUTPUTS_DIR = original_root / "outputs"
        paths_module.CHECKPOINTS_DIR = paths_module.OUTPUTS_DIR / "checkpoints"
        paths_module.METRICS_DIR = paths_module.OUTPUTS_DIR / "metrics"
        paths_module.LOGS_DIR = paths_module.OUTPUTS_DIR / "logs"
        paths_module.FIGURES_DIR = paths_module.OUTPUTS_DIR / "figures"
        paths_module.PREDICTIONS_DIR = paths_module.OUTPUTS_DIR / "predictions"

    def test_get_paths_with_fold(self, temp_project):
        """Path generation with fold parameter."""
        project_root, config_file = temp_project
        
        # Create fold directory structure
        dataset_dir = project_root / "Data" / "test_dataset"
        fold_dir = dataset_dir / "fold0"
        fold_dir.mkdir()
        (fold_dir / "train").mkdir()
        (fold_dir / "val").mkdir()
        
        paths = get_paths(config_path=config_file, fold="fold0")
        
        assert paths["FOLD_DIR"] == dataset_dir / "fold0"
        assert paths["TRAIN_DIR"] == fold_dir / "train"
        assert paths["VAL_DIR"] == fold_dir / "val"
        assert paths["TEST_DIR"] == dataset_dir / "test"
        assert paths["DATASET_DIR"] == dataset_dir

    def test_get_paths_without_fold(self, temp_project):
        """Path generation for test set."""
        project_root, config_file = temp_project
        
        paths = get_paths(config_path=config_file, fold=None)
        
        assert paths["FOLD_DIR"] is None
        assert paths["TRAIN_DIR"] is None
        assert paths["VAL_DIR"] is None
        assert paths["TEST_DIR"] == project_root / "Data" / "test_dataset" / "test"

    def test_get_paths_with_model_name(self, temp_project):
        """Model checkpoint paths."""
        project_root, config_file = temp_project
        
        # Create fold structure
        dataset_dir = project_root / "Data" / "test_dataset"
        fold_dir = dataset_dir / "fold0"
        fold_dir.mkdir()
        
        paths = get_paths(
            config_path=config_file, 
            fold="fold0", 
            model_name="test_model"
        )
        
        assert paths["MODEL_CHECKPOINT_PATH"] == project_root / "outputs" / "checkpoints" / "fold0" / "test_model.pt"
        assert paths["LOSS_ACC_PLOT_PATH"] == project_root / "outputs" / "figures" / "fold0" / "test_model_loss_plot.png"
        
        # Verify directories were created
        assert (project_root / "outputs" / "checkpoints" / "fold0").exists()
        assert (project_root / "outputs" / "figures" / "fold0").exists()

    def test_get_paths_directory_creation(self, temp_project):
        """Required directories created."""
        project_root, config_file = temp_project
        
        # Remove outputs if exists
        outputs_dir = project_root / "outputs"
        if outputs_dir.exists():
            shutil.rmtree(outputs_dir)
        
        paths = get_paths(config_path=config_file)
        
        # Verify all required directories exist
        # Note: get_paths returns paths relative to the monkey-patched PROJECT_ROOT
        assert paths["CHECKPOINTS_DIR"].exists()
        assert paths["METRICS_DIR"].exists()
        assert paths["LOGS_DIR"].exists()
        assert paths["FIGURES_DIR"].exists()
        assert paths["PREDICTIONS_DIR"].exists()


class TestCleanOutputsDir:
    """Test suite for clean_outputs_dir function."""

    def test_clean_outputs_dir(self, tmp_path):
        """Output directory cleanup."""
        # Create temporary outputs directory
        import src.utils.paths as paths_module
        original_outputs = paths_module.OUTPUTS_DIR
        
        temp_outputs = tmp_path / "outputs"
        temp_outputs.mkdir()
        
        # Create some files and directories
        (temp_outputs / "file1.txt").write_text("test")
        (temp_outputs / "file2.txt").write_text("test")
        subdir = temp_outputs / "subdir"
        subdir.mkdir()
        (subdir / "file3.txt").write_text("test")
        
        paths_module.OUTPUTS_DIR = temp_outputs
        
        clean_outputs_dir()
        
        # Verify directory is empty
        assert temp_outputs.exists()
        assert list(temp_outputs.iterdir()) == []
        
        paths_module.OUTPUTS_DIR = original_outputs

    def test_clean_outputs_dir_nonexistent(self, tmp_path):
        """Handle missing directory."""
        import src.utils.paths as paths_module
        original_outputs = paths_module.OUTPUTS_DIR
        
        # Set to non-existent directory
        nonexistent = tmp_path / "nonexistent"
        paths_module.OUTPUTS_DIR = nonexistent
        
        # Should not raise error
        clean_outputs_dir()
        
        # Directory should not exist
        assert not nonexistent.exists()
        
        paths_module.OUTPUTS_DIR = original_outputs

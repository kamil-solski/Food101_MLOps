"""
Unit tests for src.evaluation.plots module.
"""
import pytest
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import sys
from pathlib import Path
import torch
from torch.utils.data import DataLoader, TensorDataset

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.evaluation.plots import loss_acc_plot, plot_roc_ovr
from src.models.food101 import Food101


class TestLossAccPlot:
    """Test suite for loss_acc_plot function."""
    
    def test_loss_acc_plot_creation(self):
        """Plot generation."""
        results = {
            "train_loss": [0.8, 0.6, 0.4, 0.3, 0.2],
            "train_acc": [0.2, 0.4, 0.6, 0.7, 0.8],
            "val_loss": [0.9, 0.7, 0.5, 0.4, 0.35],
            "val_acc": [0.15, 0.35, 0.55, 0.65, 0.75]
        }
        
        fig = loss_acc_plot(results)
        
        assert fig is not None
        assert hasattr(fig, "axes")
        assert len(fig.axes) == 2  # Two subplots

    def test_loss_acc_plot_figure_properties(self):
        """Figure properties."""
        results = {
            "train_loss": [0.8, 0.6, 0.4],
            "train_acc": [0.2, 0.4, 0.6],
            "val_loss": [0.9, 0.7, 0.5],
            "val_acc": [0.15, 0.35, 0.55]
        }
        
        fig = loss_acc_plot(results)
        
        assert fig.get_figwidth() > 0
        assert fig.get_figheight() > 0
        assert len(fig.axes) == 2


class TestPlotRocOvr:
    """Test suite for plot_roc_ovr function."""
    
    @pytest.fixture
    def mock_model_dict(self):
        """Create mock model dictionary."""
        model1 = Food101(input_shape=3, hidden_units=8, output_shape=4)
        model2 = Food101(input_shape=3, hidden_units=16, output_shape=4)
        
        return {
            "Food101_hu8": {
                "model": model1,
                "run_id": "run1",
                "hu": 8,
                "lr": 0.001
            },
            "Food101_hu16": {
                "model": model2,
                "run_id": "run2",
                "hu": 16,
                "lr": 0.0001
            }
        }
    
    @pytest.fixture
    def mock_dataloader(self):
        """Create mock dataloader."""
        X = torch.randn(10, 3, 64, 64)
        y = torch.randint(0, 4, (10,))
        dataset = TensorDataset(X, y)
        return DataLoader(dataset, batch_size=5, shuffle=False)

    @pytest.mark.skip(reason="Requires sklearn and proper evaluation setup")
    def test_plot_roc_ovr_creation(self, mock_model_dict, mock_dataloader):
        """ROC plot generation."""
        class_names = ["class0", "class1", "class2", "class3"]
        
        fig, auc_scores = plot_roc_ovr(mock_model_dict, mock_dataloader, class_names)
        
        assert fig is not None
        assert isinstance(auc_scores, dict)
        assert len(auc_scores) == len(class_names)

    @pytest.mark.skip(reason="Requires sklearn and proper evaluation setup")
    def test_plot_roc_ovr_auc_calculation(self, mock_model_dict, mock_dataloader):
        """AUC computation."""
        class_names = ["class0", "class1", "class2", "class3"]
        
        _, auc_scores = plot_roc_ovr(mock_model_dict, mock_dataloader, class_names)
        
        for class_name, model_aucs in auc_scores.items():
            assert class_name in class_names
            for model_name, meta in model_aucs.items():
                assert "auc" in meta
                assert 0 <= meta["auc"] <= 1

    @pytest.mark.skip(reason="Requires sklearn and proper evaluation setup")
    def test_plot_roc_ovr_multiple_models(self, mock_model_dict, mock_dataloader):
        """Multiple model comparison."""
        class_names = ["class0", "class1", "class2", "class3"]
        
        _, auc_scores = plot_roc_ovr(mock_model_dict, mock_dataloader, class_names)
        
        # All models should be in AUC scores for each class
        for class_name in class_names:
            assert len(auc_scores[class_name]) == len(mock_model_dict)

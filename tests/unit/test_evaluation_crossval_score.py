"""
Unit tests for src.evaluation.crossval_score module.
NOTE: These tests require MLflow tracking server or mocked MLflow.
They are placeholders and should be expanded with proper mocking.
"""
import pytest
import pandas as pd
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.evaluation.crossval_score import (
    filter_by_loss_discrepancy,
    group_by_arch_and_config,
    score_models,
    select_best_configs,
    select_best_model_from_auc
)


class TestFilterByLossDiscrepancy:
    """Test suite for filter_by_loss_discrepancy function."""
    
    def test_filter_by_loss_discrepancy(self):
        """Filtering logic."""
        # Create mock runs DataFrame
        runs = pd.DataFrame({
            "metrics.train_loss": [0.5, 0.6, 0.7],
            "metrics.val_loss": [0.4, 1.0, 0.5],  # Second one: |0.6 - 1.0| = 0.4 > 0.25, should be filtered
        })
        
        filtered = filter_by_loss_discrepancy(runs, threshold=0.25)
        
        assert len(filtered) < len(runs)  # Should filter out at least one
        assert len(filtered) == 2  # Should keep only rows 0 and 2
        assert "loss_diff" in filtered.columns
        
        # All remaining should have discrepancy <= threshold
        assert all(filtered["loss_diff"] <= 0.25)


class TestGroupByArchAndConfig:
    """Test suite for group_by_arch_and_config function."""
    
    def test_group_by_arch_and_config(self):
        """Grouping operations."""
        runs = pd.DataFrame({
            "tags.architecture": ["Food101", "Food101", "Food101_0", "Food101_0"],
            "tags.config": ["hu=8,lr=0.001", "hu=8,lr=0.001", "hu=16,lr=0.0001", "hu=16,lr=0.0001"],
            "metrics.val_loss": [0.5, 0.6, 0.7, 0.8],
            "metrics.val_acc": [0.8, 0.75, 0.7, 0.65]
        })
        
        grouped = group_by_arch_and_config(runs, val_metric="val_acc")
        
        assert isinstance(grouped, pd.DataFrame)
        assert "architecture" in grouped.columns
        assert "config" in grouped.columns
        assert "avg_val_acc" in grouped.columns


class TestScoreModels:
    """Test suite for score_models function."""
    
    def test_score_models(self):
        """Scoring calculations."""
        df = pd.DataFrame({
            "architecture": ["Food101", "Food101_0"],
            "avg_val_loss": [0.5, 0.7],
            "avg_val_acc": [0.8, 0.6]
        })
        
        scored = score_models(df, val_metric="val_acc", acc_weight=0.7, loss_weight=0.3)
        
        assert "score" in scored.columns
        assert len(scored) == len(df)
        # Better model should have higher score
        assert scored.iloc[0]["score"] > scored.iloc[1]["score"]


class TestSelectBestConfigs:
    """Test suite for select_best_configs function."""
    
    def test_select_best_configs(self):
        """Best config selection."""
        df_scored = pd.DataFrame({
            "architecture": ["Food101", "Food101", "Food101_0", "Food101_0"],
            "score": [0.9, 0.7, 0.8, 0.6]
        })
        
        best = select_best_configs(df_scored)
        
        # Should have one config per architecture
        assert len(best) == 2
        assert len(best["architecture"].unique()) == 2


class TestSelectBestModelFromAuc:
    """Test suite for select_best_model_from_auc function."""
    
    def test_select_best_model_from_auc(self):
        """AUC-based selection."""
        auc_scores = {
            "class1": {
                "model1": {"auc": 0.9, "run_id": "run1"},
                "model2": {"auc": 0.8, "run_id": "run2"}
            },
            "class2": {
                "model1": {"auc": 0.85, "run_id": "run1"},
                "model2": {"auc": 0.75, "run_id": "run2"}
            }
        }
        
        best_model, model_avg_auc = select_best_model_from_auc(auc_scores)
        
        assert isinstance(best_model, dict)
        assert "model1" in best_model  # Should have highest average AUC
        assert "model1" in model_avg_auc


@pytest.mark.skip(reason="Requires MLflow experiment setup")
class TestGetAllRuns:
    """Test suite for get_all_runs function."""
    
    def test_get_all_runs_missing_experiment(self):
        """Error handling."""
        # TODO: Implement with mocked MLflow
        from src.evaluation.crossval_score import get_all_runs
        
        with pytest.raises(ValueError, match="not found"):
            get_all_runs("nonexistent_experiment")

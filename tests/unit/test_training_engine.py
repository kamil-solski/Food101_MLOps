"""
Unit tests for src.training.engine module.
"""
import pytest
import torch
import torch.nn as nn
import numpy as np
import sys
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.training.engine import PyTorchTrainer
from src.models.food101 import Food101


@pytest.fixture
def mock_dataloader_4class():
    X = torch.randn(10, 3, 64, 64)
    y = torch.randint(0, 4, (10,))
    return DataLoader(TensorDataset(X, y), batch_size=4, shuffle=False)


@pytest.fixture
def base_trainer(mock_dataloader_4class, device):
    model = Food101(input_shape=3, hidden_units=8, output_shape=4).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    return PyTorchTrainer(model, loss_fn, optimizer, device)


class TestTrainStep:
    def test_returns_float_metrics(self, base_trainer, mock_dataloader_4class):
        train_loss, train_acc = base_trainer.train_step(mock_dataloader_4class)
        assert isinstance(train_loss, float)
        assert isinstance(train_acc, float)
        assert train_loss >= 0
        assert 0 <= train_acc <= 1

    def test_no_nan(self, base_trainer, mock_dataloader_4class):
        loss1, _ = base_trainer.train_step(mock_dataloader_4class)
        loss2, _ = base_trainer.train_step(mock_dataloader_4class)
        assert not np.isnan(loss1)
        assert not np.isnan(loss2)

    def test_gradients_computed(self, base_trainer, mock_dataloader_4class):
        base_trainer.optimizer.zero_grad()
        base_trainer.train_step(mock_dataloader_4class)
        has_gradients = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in base_trainer.model.parameters()
        )
        assert has_gradients


class TestEvalStep:
    @pytest.fixture
    def val_dataloader(self):
        X = torch.randn(8, 3, 64, 64)
        y = torch.randint(0, 4, (8,))
        return DataLoader(TensorDataset(X, y), batch_size=4, shuffle=False)

    def test_returns_float_metrics(self, base_trainer, val_dataloader):
        val_loss, val_acc = base_trainer.eval_step(val_dataloader)
        assert isinstance(val_loss, float)
        assert isinstance(val_acc, float)
        assert val_loss >= 0
        assert 0 <= val_acc <= 1

    def test_no_gradients_computed(self, base_trainer, val_dataloader):
        for param in base_trainer.model.parameters():
            param.grad = None
        base_trainer.eval_step(val_dataloader)
        for param in base_trainer.model.parameters():
            assert param.grad is None


class TestGetPredictions:
    @pytest.fixture
    def pred_dataloader(self):
        X = torch.randn(6, 3, 64, 64)
        y = torch.randint(0, 4, (6,))
        return DataLoader(TensorDataset(X, y), batch_size=3, shuffle=False)

    def test_output_types(self, base_trainer, pred_dataloader):
        y_true, y_probs = base_trainer.get_predictions(pred_dataloader)
        assert isinstance(y_true, np.ndarray)
        assert isinstance(y_probs, np.ndarray)

    def test_output_shapes(self, base_trainer, pred_dataloader):
        y_true, y_probs = base_trainer.get_predictions(pred_dataloader)
        assert y_true.shape == (6,)
        assert y_probs.shape == (6, 4)

    def test_probs_sum_to_one(self, base_trainer, pred_dataloader):
        _, y_probs = base_trainer.get_predictions(pred_dataloader)
        assert np.allclose(y_probs.sum(axis=1), 1.0, atol=1e-6)
        assert np.all(y_probs >= 0)

"""
Integration tests for the training pipeline.

These tests exercise multiple real components together (data loading → model →
trainer → MLflow) using a tiny synthetic dataset so the suite stays fast.
Run with: pytest tests/integration/ -m integration
"""
import sys
from pathlib import Path

import mlflow
import pytest
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.dataloader import get_dataloaders
from src.models.food101 import Food101
from src.models.food101_0 import Food101_0
from src.training.engine import PyTorchTrainer


# ---------------------------------------------------------------------------
# Data-loading pipeline
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestDataLoadingPipeline:
    """Real ImageFolder loading from synthetic images."""

    def test_fold_dataloaders_return_correct_shapes(self, patched_paths):
        train_loader, val_loader, class_names = get_dataloaders(
            image_size=64, batch_size=4, fold="fold0"
        )

        assert len(class_names) == 2  # apple, banana

        X, y = next(iter(train_loader))
        assert X.shape == (4, 3, 64, 64)
        assert y.shape == (4,)
        assert X.dtype == torch.float32

    def test_pixel_values_in_unit_range(self, patched_paths):
        train_loader, _, _ = get_dataloaders(
            image_size=64, batch_size=4, fold="fold0"
        )
        X, _ = next(iter(train_loader))
        assert X.min() >= 0.0
        assert X.max() <= 1.0

    def test_test_loader_without_fold(self, patched_paths):
        test_loader, class_names = get_dataloaders(
            image_size=64, batch_size=4, fold=None, mode="test"
        )
        assert len(class_names) == 2
        X, _ = next(iter(test_loader))
        assert X.shape[1:] == (3, 64, 64)

    def test_class_names_match_classes_txt(self, patched_paths):
        _, _, class_names = get_dataloaders(
            image_size=64, batch_size=4, fold="fold0"
        )
        expected = sorted(["apple", "banana"])
        assert sorted(class_names) == expected


# ---------------------------------------------------------------------------
# Trainer integration
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestTrainerWithRealData:
    """PyTorchTrainer running on synthetic batches from ImageFolder."""

    @pytest.fixture
    def loaders_and_model(self, patched_paths):
        train_loader, val_loader, class_names = get_dataloaders(
            image_size=64, batch_size=4, fold="fold0"
        )
        model = Food101(input_shape=3, hidden_units=8, output_shape=len(class_names))
        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        trainer = PyTorchTrainer(model, loss_fn, optimizer)
        return trainer, train_loader, val_loader, class_names

    def test_train_step_returns_valid_metrics(self, loaders_and_model):
        trainer, train_loader, _, _ = loaders_and_model
        train_loss, train_acc = trainer.train_step(train_loader)

        assert isinstance(train_loss, float)
        assert isinstance(train_acc, float)
        assert train_loss >= 0.0
        assert 0.0 <= train_acc <= 1.0
        assert not (train_loss != train_loss)  # NaN check

    def test_eval_step_returns_valid_metrics(self, loaders_and_model):
        trainer, _, val_loader, _ = loaders_and_model
        val_loss, val_acc = trainer.eval_step(val_loader)

        assert isinstance(val_loss, float)
        assert 0.0 <= val_acc <= 1.0

    def test_weights_update_after_train_step(self, loaders_and_model):
        trainer, train_loader, _, _ = loaders_and_model
        before = [p.clone().detach() for p in trainer.model.parameters()]
        trainer.train_step(train_loader)
        after = list(trainer.model.parameters())

        changed = any(not torch.equal(b, a) for b, a in zip(before, after))
        assert changed, "No parameter updated after a training step"

    def test_eval_does_not_change_weights(self, loaders_and_model):
        trainer, _, val_loader, _ = loaders_and_model
        before = [p.clone().detach() for p in trainer.model.parameters()]
        trainer.eval_step(val_loader)
        after = list(trainer.model.parameters())

        assert all(torch.equal(b, a) for b, a in zip(before, after))

    def test_get_predictions_shapes(self, loaders_and_model):
        import numpy as np
        trainer, _, val_loader, class_names = loaders_and_model
        y_true, y_probs = trainer.get_predictions(val_loader)

        n_samples = sum(len(y) for _, y in val_loader)
        assert y_true.shape == (n_samples,)
        assert y_probs.shape == (n_samples, len(class_names))
        assert np.allclose(y_probs.sum(axis=1), 1.0, atol=1e-5)


# ---------------------------------------------------------------------------
# MLflow integration
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestMLflowTraining:
    """train_with_mlflow logs metrics and saves a checkpoint."""

    def test_full_training_run_logs_metrics(self, patched_paths, tmp_path):
        from src.training.train import train_with_mlflow
        from src.utils.paths import get_paths

        mlflow_dir = tmp_path / "mlruns"
        mlflow.set_tracking_uri(mlflow_dir.as_uri())

        train_loader, val_loader, class_names = get_dataloaders(
            image_size=64, batch_size=4, fold="fold0"
        )
        model = Food101(input_shape=3, hidden_units=8, output_shape=len(class_names))
        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        paths = get_paths(
            config_path=patched_paths["config_file"],
            fold="fold0",
            model_name="test_model",
        )

        mlflow.set_experiment("test_experiment")
        with mlflow.start_run():
            results = train_with_mlflow(
                model=model,
                model_name="test_model",
                train_dataloader=train_loader,
                val_dataloader=val_loader,
                loss_fn=loss_fn,
                optimizer=optimizer,
                learning_rate=1e-3,
                epochs=1,
                paths=paths,
            )

        assert "train_loss" in results
        assert "val_acc" in results
        assert len(results["train_loss"]) == 1
        assert 0.0 <= results["val_acc"][0] <= 1.0

    def test_checkpoint_saved_after_training(self, patched_paths, tmp_path):
        from src.training.train import train_with_mlflow
        from src.utils.paths import get_paths

        mlflow_dir = tmp_path / "mlruns"
        mlflow.set_tracking_uri(mlflow_dir.as_uri())

        train_loader, val_loader, class_names = get_dataloaders(
            image_size=64, batch_size=4, fold="fold0"
        )
        model = Food101(input_shape=3, hidden_units=8, output_shape=len(class_names))
        paths = get_paths(
            config_path=patched_paths["config_file"],
            fold="fold0",
            model_name="ckpt_test",
        )

        mlflow.set_experiment("test_checkpoint")
        with mlflow.start_run():
            train_with_mlflow(
                model=model,
                model_name="ckpt_test",
                train_dataloader=train_loader,
                val_dataloader=val_loader,
                loss_fn=nn.CrossEntropyLoss(),
                optimizer=torch.optim.Adam(model.parameters(), lr=1e-3),
                learning_rate=1e-3,
                epochs=1,
                paths=paths,
            )

        assert paths["MODEL_CHECKPOINT_PATH"].exists()


# ---------------------------------------------------------------------------
# Both architectures
# ---------------------------------------------------------------------------

@pytest.mark.integration
@pytest.mark.parametrize("ModelClass", [Food101, Food101_0])
def test_architecture_trains_one_epoch(ModelClass, patched_paths):
    """Both CNN architectures complete a training epoch without errors."""
    train_loader, val_loader, class_names = get_dataloaders(
        image_size=64, batch_size=4, fold="fold0"
    )
    model = ModelClass(input_shape=3, hidden_units=8, output_shape=len(class_names))
    trainer = PyTorchTrainer(
        model,
        nn.CrossEntropyLoss(),
        torch.optim.Adam(model.parameters(), lr=1e-3),
    )

    train_loss, train_acc = trainer.train_step(train_loader)
    val_loss, val_acc = trainer.eval_step(val_loader)

    assert train_loss >= 0.0
    assert 0.0 <= val_acc <= 1.0

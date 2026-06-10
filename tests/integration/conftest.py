"""
Shared fixtures for integration tests.
Reuses the session-scoped `synthetic_dataset` from the root conftest.
"""
import yaml
import pytest


@pytest.fixture
def patched_paths(synthetic_dataset, tmp_path):
    """
    Redirect all paths-module globals to the synthetic dataset and a temp
    outputs directory.  Restores originals after the test.

    Yields a dict:
        {"config_file": Path, "dataset_dir": Path, "outputs_dir": Path}
    """
    import src.utils.paths as pm
    import src.data.dataloader as dm

    config_file = tmp_path / "config.yaml"
    with open(config_file, "w") as f:
        yaml.dump(
            {
                "dataset": synthetic_dataset.name,
                "batch_size": 4,
                "image_size": 64,
                "num_epochs": 1,
                "learning_rates": [0.001],
                "hidden_units": [8],
                "architectures": ["Food101"],
                "random_seed": 42,
            },
            f,
        )

    saved = {k: getattr(pm, k) for k in (
        "DATA_DIR", "OUTPUTS_DIR", "CHECKPOINTS_DIR",
        "METRICS_DIR", "LOGS_DIR", "FIGURES_DIR", "PREDICTIONS_DIR",
    )}
    saved["get_paths"] = pm.get_paths

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

    yield {
        "config_file": config_file,
        "dataset_dir": synthetic_dataset,
        "outputs_dir": tmp_path / "outputs",
    }

    for k, v in saved.items():
        if k == "get_paths":
            pm.get_paths = v
            dm.get_paths = v
        else:
            setattr(pm, k, v)

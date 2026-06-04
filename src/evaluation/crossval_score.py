"""
Tags are essential in our use case (especially architecture and fold) bacasuse they allow us to filter neccessary runs to extract the best hyperparameter combination from folds.
Because we got folds we need to get averages per hyperparameter combination across folds (for each architecture).
So, from num_architectures * num_folds * num_combination we should get num_architectures of models.
We will filter by tags, averege per combination and pick the highest value for each architecture.

It is better to load models from mlruns than temporary because it gives us more options (we are not limited to only current run). In cli.py we could modify script to choose the best models from multiple experiments
"""
import mlflow
import pandas as pd
from collections import defaultdict
import numpy as np

# This function was directly copied from notebooks/Manually_pick_best_model.ipynb
def get_all_runs(experiment_name: str) -> pd.DataFrame:
    experiment = mlflow.get_experiment_by_name(experiment_name)  # Get experiment by name
    if experiment is None:
        raise ValueError(f"Experiment '{experiment_name}' not found.")
    
    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],  # Search within this experiment
        filter_string="attributes.status = 'FINISHED'",  # Only finished runs
        output_format="pandas"  # Return as pandas dataframe
    )
    return runs


def filter_by_loss_discrepancy(runs: pd.DataFrame, threshold: float = 0.25) -> pd.DataFrame:
    """
    Remove runs that overfit
    Args:
        runs: pd.DataFrame - runs dataframe
        threshold: float - threshold for overfitting
    Returns:
        pd.DataFrame - runs dataframe
    """
    runs["loss_diff"] = (runs["metrics.train_loss"] - runs["metrics.val_loss"]).abs()  # calculate overfitting
    return runs[runs["loss_diff"] <= threshold].copy()  # if it is below threshold we keep it


def group_by_arch_and_config(runs: pd.DataFrame, val_metric: str = "val_acc") -> pd.DataFrame:
    """
    Group runs by architecture, configuration and take averages of combinations
    Args:
        runs: pd.DataFrame - runs dataframe
        val_metric: str - validation metric
    Returns:
        pd.DataFrame - runs dataframe
    """
    grouped = runs.groupby(["tags.architecture", "tags.config"])  # Group by architecture and configuration

    summary = []
    for (arch, config), group in grouped:  # For each arch and config combination (look Cross validation and hyperparameters automation section in README.md)
        avg_val_loss = group["metrics.val_loss"].mean()  # Calculate average validation loss
        avg_val_metric = group[f"metrics.{val_metric}"].mean()  # Calculate average validation metric (for example accuracy)

        summary.append({  # smmary for given combination
            "architecture": arch,  
            "config": config,
            "folds": len(group),
            "avg_val_loss": avg_val_loss,
            f"avg_{val_metric}": avg_val_metric
        })

    return pd.DataFrame(summary)


def score_models(df: pd.DataFrame, val_metric: str = "val_acc",
                 acc_weight: float = 0.7, loss_weight: float = 0.3) -> pd.DataFrame:
    df = df.copy()

    # Normalize scores to 0-1 range; guard against all-equal values (avoids NaN)
    loss_range = df["avg_val_loss"].max() - df["avg_val_loss"].min()
    metric_range = df[f"avg_{val_metric}"].max() - df[f"avg_{val_metric}"].min()

    df["loss_score"] = 1 - (df["avg_val_loss"] - df["avg_val_loss"].min()) / (loss_range if loss_range > 0 else 1)
    df["metric_score"] = (df[f"avg_{val_metric}"] - df[f"avg_{val_metric}"].min()) / (metric_range if metric_range > 0 else 1)

    # Weighted score
    df["score"] = acc_weight * df["metric_score"] + loss_weight * df["loss_score"]
    return df.sort_values("score", ascending=False)


def select_best_configs(df_scored: pd.DataFrame) -> pd.DataFrame:
    return df_scored.loc[df_scored.groupby("architecture")["score"].idxmax()].reset_index(drop=True) 


def load_best_models(best_config: pd.DataFrame, experiment_name: str) -> dict:
    """
    Loads best models from MLflow using best_config dataframe.
    Returns: dict of {architecture_name: model}
    """
    model_dict = {}

    for _, row in best_config.iterrows():  # For each best configuration
        arch = row["architecture"]  # Architecture name
        config_str = row["config"]  # hyperparameters combination string

        # Extract hyperparameters from string
        hu = int(config_str.split("hu=")[1].split(",")[0])
        lr = float(config_str.split("lr=")[1])

        # Find the specific run with this configuration
        experiment = mlflow.get_experiment_by_name(experiment_name)
        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string=f"tags.architecture = '{arch}' and tags.config = '{config_str}'",
            output_format="pandas"
        )
        if runs.empty:
            raise ValueError(f"No run found for arch={arch}, config={config_str}")
        
        # Use the best-scoring run (highest val_acc or final score)
        best_run = runs.sort_values("metrics.val_acc", ascending=False).iloc[0]
        run_id = best_run.run_id

        # Load model from MLflow artifact
        model = mlflow.pytorch.load_model(f"runs:/{run_id}/model")
        model.eval()  # Set model to evaluation mode because now we will evaluate it (training was done)
        model_dict[arch] = {
            "model": model,  # The actual pytorch model
            "run_id": run_id,  # mlflow run id
            # hyperparameters:
            "hu": hu,  # hidden units
            "lr": lr  # learning rate
        }

    return model_dict


def select_best_model_from_auc(auc_scores: dict) -> dict:
    """
    Because AUC just like accuracy or other can be calcualated only for binary classification (either we got right class or not), we need to again pick the higest average AUC across all classes.
    
    auc_scores: dict[class_name][model_name] = {"auc": float, "run_id": str}

    Returns:
        best_model_dict: {model_full_name: run_id}
        model_avg_auc: {model_full_name: average_auc}
    """
    model_totals = defaultdict(list)
    model_run_ids = {}
    
    # Collect AUC scores for each model across all classes
    for class_aucs in auc_scores.values():  # For each class
        for model_name, data in class_aucs.items():  # For each model
            model_totals[model_name].append(data["auc"])
            model_run_ids[model_name] = data["run_id"]

    # Calculate average
    model_avg_auc = {
        model: np.mean(scores) for model, scores in model_totals.items()
    }

    # Pick the model with the highest average AUC
    best_model_name = max(model_avg_auc, key=model_avg_auc.get)
    best_run_id = model_run_ids[best_model_name]

    best_model_dict = {best_model_name: best_run_id}

    return best_model_dict, model_avg_auc
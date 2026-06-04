import yaml
import mlflow
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
import matplotlib.pyplot as plt
import gc
import random
import numpy as np

from src.utils.helpers import set_seed
from src.evaluation import loss_acc_plot, plot_roc_ovr
from src.models import architectures
from src.data.dataloader import get_dataloaders
from src.training.train import train_with_mlflow
from src.utils.paths import DATA_DIR, get_paths, MLFLOW_TRACKING_DIR, clean_outputs_dir, FIGURES_DIR
from src.serving.onnx_registry import ensure_onnx_and_register
from src.evaluation.crossval_score import (
    get_all_runs,
    filter_by_loss_discrepancy,
    group_by_arch_and_config,
    score_models,
    select_best_configs,
    load_best_models,
    select_best_model_from_auc
)

MLFLOW_TRACKING_DIR.mkdir(parents=True, exist_ok=True)
mlflow.set_tracking_uri(MLFLOW_TRACKING_DIR.as_uri())


def main():
    # Data and config loading
    with open("src/config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    # Set random seed for reproducibility
    set_seed(seed=config.get("random_seed", 42))

    dataset_name = config["dataset"]
    data_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    experiment_name = f"{dataset_name}_{data_str}"  # with this it should be used strategically (with changing hyperparameters for each experiment)
    
    mlflow.set_experiment(experiment_name)
    
    # Log experiment-level metadata for reproducibility
    with mlflow.start_run(run_name="experiment_metadata"):
        mlflow.set_tag("random_seed", config.get("random_seed", 42))
        mlflow.set_tag("reproducible", "true")
        mlflow.log_param("config_file", "src/config.yaml")
    mlflow.end_run()
    
    dataset_path = DATA_DIR / dataset_name
    
    # Auto-detect dataset structure: cross-validation folds or simple train/val/test
    if any(d.name.startswith("fold") for d in dataset_path.iterdir() if d.is_dir()):
        # Cross-validation mode
        folds = sorted([d for d in dataset_path.iterdir() if d.is_dir() and d.name.startswith("fold")])
        print(f"[INFO] Detected {len(folds)} folds for cross-validation")
        use_cv = True
    else:
        # Simple train/val/test mode (treat as single iteration, no folds)
        folds = [None]
        print(f"[INFO] Single train/val/test split detected (no cross-validation)")
        use_cv = False

    # Handle both cross-validation and simple splits with optimized loop order
    # Loop order: architectures -> hyperparameters -> folds (innermost)
    # This allows better memory management and logical grouping of experiments
    for arch_name in config["architectures"]:
        model_class = architectures[arch_name]
        for hu in config["hidden_units"]:
            for lr in config["learning_rates"]:
                if use_cv:
                    print(f"\n[INFO] Training {arch_name} with hu={hu}, lr={lr} across all folds")
                else:
                    print(f"\n[INFO] Training {arch_name} with hu={hu}, lr={lr}")
                
                # Train this hyperparameter combination across all folds (or once if no CV)
                for fold in folds:
                    if use_cv:
                        print(f"  [INFO] Running fold: {fold.name}")
                    else:
                        print(f"  [INFO] Running training on single split")

                    # Create dataloaders (works for both CV and simple splits)
                    train_loader, val_loader, class_names = get_dataloaders(
                        image_size=config["image_size"],
                        batch_size=config["batch_size"],
                        fold=fold,  # Path object (CV mode) or None (simple mode)
                        mode="train"  # Explicitly request train/val loaders
                    )

                    model = model_class(input_shape=3, hidden_units=hu, output_shape=len(class_names), image_size=config["image_size"])
                    model = model.to("cuda" if torch.cuda.is_available() else "cpu")

                    model_name = f"{arch_name}_hu{hu}_lr{lr}"
                    
                    # Determine paths and run name based on mode
                    if use_cv:
                        run_name = f"{model_name}_{fold.name}"
                        paths = get_paths(fold=fold.name, model_name=model_name)
                        fold_tag = fold.name
                    else:
                        run_name = model_name
                        paths = get_paths(fold=None, model_name=model_name)
                        fold_tag = "single_split"
 
                    with mlflow.start_run(run_name=run_name):
                        mlflow.set_tag("fold", fold_tag)
                        mlflow.set_tag("architecture", arch_name)
                        mlflow.set_tag("config", f"hu={hu},lr={lr}")                    
                    
                        results = train_with_mlflow(
                            model=model,
                            model_name=model_name,
                            train_dataloader=train_loader,
                            val_dataloader=val_loader,
                            loss_fn=nn.CrossEntropyLoss(),
                            optimizer=optim.Adam(model.parameters(), lr=lr),
                            learning_rate=lr,
                            epochs=config["num_epochs"],
                            paths=paths
                        )
                        
                        # Save training curve as image
                        fig = loss_acc_plot(results)
                        fig.savefig(paths["LOSS_ACC_PLOT_PATH"])
                        
                        # Log to MLflow with proper artifact path
                        mlflow.log_figure(fig, "figures/training_plot.png")
                        plt.close(fig)  # prevent memory leak
                        
                        # Cleanup after each model training
                        del model
                        gc.collect()
                        torch.cuda.empty_cache()
                
                # Cleanup after completing all folds for this hyperparameter combination
                gc.collect()
                torch.cuda.empty_cache()
    
    # get the best models for each architecture from cross-validation
    print("\n[INFO] Running automatic model selection from cross-validation results...")
    
    all_runs = get_all_runs(experiment_name=experiment_name)  # current experiment
    filtered_runs = filter_by_loss_discrepancy(runs=all_runs, threshold=0.25)
    grouped_runs = group_by_arch_and_config(filtered_runs, val_metric="val_acc")  # only validation accuracy matter. Training accuracy might be misleading when we would use batch normalization or dropout
    scored = score_models(grouped_runs, val_metric="val_acc", acc_weight=0.7, loss_weight=0.3)  # you can modify weights based on how important you think those metrics sould be
    best_config = select_best_configs(scored)
    
    # Loading unified dataset for testing
    test_loader, class_names = get_dataloaders( 
        image_size=config["image_size"],
        batch_size=config["batch_size"],
        fold=None,  # Use dataset root structure
        mode="test"  # Explicitly request test loader
    )
    
    # Loading models from crossval_score.py
    model_dict = load_best_models(best_config=best_config, experiment_name=experiment_name)
    
    #print(model_dict)
    
    # Plot ROCs per class for all models
    fig, auc_scores = plot_roc_ovr(
        model_dict=model_dict,
        test_dataloader=test_loader,
        class_names=class_names
    )

    # Log ROC plot to MLflow as experiment-level artifact
    print(f"[INFO] Logging ROC analysis to MLflow for experiment: {experiment_name}")
    with mlflow.start_run(run_name="experiment_roc_analysis"):
        mlflow.log_figure(fig, "figures/roc_per_class.png")
        
        # Log AUC metrics for each class and model
        for cls, model_aucs in auc_scores.items():
            for model_name, meta in model_aucs.items():
                mlflow.log_metric(f"AUC_{cls}_{model_name}", meta["auc"])
        
        # Log summary metrics
        mlflow.set_tag("analysis_type", "roc_evaluation")
        mlflow.set_tag("total_classes", len(class_names))
        mlflow.set_tag("total_models", len(model_dict))
        print(f"[INFO] Successfully logged ROC analysis with {len(class_names)} classes and {len(model_dict)} models")

    plt.close(fig)  # prevent memory leak
    
    # Cleanup after final evaluation
    del test_loader, model_dict, fig
    gc.collect()
    torch.cuda.empty_cache()
    
    best_model, _ = select_best_model_from_auc(auc_scores=auc_scores)  # we could save to temp file best score auc and during next experiment (doesn't matter how it would be called) compare recent current with previous and decide using client if we should update to production
    
    # Register best model (no automatic alias/promotion).
    # Promotion is a deliberate human decision: open the MLflow UI and set the
    # 'champion' alias on the chosen version (Models -> best_model -> version ->
    # add alias 'champion'). The serving API loads ONLY the 'champion' version and
    # refuses to start if no champion is set, so remember to promote one before serving.
    print(f"\n[INFO] Registering best model...")

    model_registry_name = "best_model"

    for model_name, run_id in best_model.items():
        # ensure ONNX exists for this run and register that ONNX
        version = ensure_onnx_and_register(
            run_id=run_id,
            registry_name=model_registry_name,
            image_size=config["image_size"],         # match training input
            class_names=class_names,
            input_name="images",                     # keep consistent with export & inference
            output_name="logits",
            opset=13,
            await_registration_for=300
        )
        print(f"[INFO] Registered ONNX for '{model_name}' as version {version} in '{model_registry_name}'")

    print("[INFO] To serve a model, set the 'champion' alias on the desired "
          "version in the MLflow UI. Inference serves ONLY the champion and will "
          "error on startup if none is set.")


if __name__ == "__main__":
    clean_outputs_dir()
    main()

# run experiments:
# cd Food101_MLOps-Lvl_1
# PYTHONPATH=. python src/cli.py
# mlflow ui --backend-store-uri experiments/mlruns
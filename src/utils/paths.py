from pathlib import Path
import yaml
import shutil

# Automatically get project root (2 levels up from this file)
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Core folders
DATA_DIR = PROJECT_ROOT / "Data"  # required to exist else raise error
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
CHECKPOINTS_DIR = OUTPUTS_DIR / "checkpoints"
METRICS_DIR = OUTPUTS_DIR / "metrics"
LOGS_DIR = OUTPUTS_DIR / "logs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
PREDICTIONS_DIR = OUTPUTS_DIR / "predictions"
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"  # created automatically in cli.py before get_paths()
MLFLOW_TRACKING_DIR = EXPERIMENTS_DIR / "mlruns"  # created automatically

def get_paths(config_path=PROJECT_ROOT / "src" / "config.yaml", fold=None, model_name=None):
    """
    Get paths for dataset and outputs.
    
    Args:
        config_path: Path to config.yaml
        fold: Can be one of:
            - Path object (e.g., Path("Data/Food101/fold_1")) → cross-validation mode
            - str (e.g., "fold_1") → cross-validation mode
            - None → simple train/val/test mode (no folds)
        model_name: Name of the model for checkpoint/figure paths
    
    Returns:
        Dictionary with all relevant paths
    """
    # Load dataset name from config
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    dataset_folder = config["dataset"]
    dataset_dir = DATA_DIR / dataset_folder
    classes_file = dataset_dir / "classes.txt"

    test_dir = dataset_dir / "test"  # global heldout set
    fold_dir = None
    fold_name = None
    train_dir = None
    val_dir = None

    # Dataset split paths
    if fold is not None:
        # Cross-validation mode: train/val inside fold directory
        if isinstance(fold, Path):
            fold_dir = fold
            fold_name = fold.name
        else:
            fold_dir = dataset_dir / fold
            fold_name = fold
        
        train_dir = fold_dir / "train"
        val_dir = fold_dir / "val"
    else:
        # Simple mode: train/val at dataset root level
        train_dir = dataset_dir / "train"
        val_dir = dataset_dir / "val"
        fold_dir = None
        fold_name = None

    # These folders are required becuase algorithm will put some temoporary results here. Before new experiment run OUTPUTS_DIR will be cleaned
    required_dirs = [ 
        OUTPUTS_DIR, CHECKPOINTS_DIR, METRICS_DIR, LOGS_DIR,
        FIGURES_DIR, PREDICTIONS_DIR 
    ]
    for d in required_dirs:
        d.mkdir(parents=True, exist_ok=True)

    # Subdirectories and file paths for model checkpoints and plots
    model_ckpt_path = None
    loss_acc_plot_path = None

    if model_name:
        if fold_name:
            # Cross-validation mode: save in fold-specific subdirectories
            model_ckpt_path = CHECKPOINTS_DIR / fold_name / f"{model_name}.pt"
            loss_acc_plot_path = FIGURES_DIR / fold_name / f"{model_name}_loss_plot.png"
            # Ensure folders exist
            (CHECKPOINTS_DIR / fold_name).mkdir(parents=True, exist_ok=True)
            (FIGURES_DIR / fold_name).mkdir(parents=True, exist_ok=True)
        else:
            # Simple mode: save directly in root
            model_ckpt_path = CHECKPOINTS_DIR / f"{model_name}.pt"
            loss_acc_plot_path = FIGURES_DIR / f"{model_name}_loss_plot.png"
            CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
            FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    return {
        "PROJECT_ROOT": PROJECT_ROOT,
        "DATA_DIR": DATA_DIR,
        "DATASET_DIR": dataset_dir,
        "FOLD_DIR": fold_dir,
        "TRAIN_DIR": train_dir,
        "VAL_DIR": val_dir,
        "TEST_DIR": test_dir,
        "CLASSES_FILE": classes_file,
        "CHECKPOINTS_DIR": CHECKPOINTS_DIR,
        "METRICS_DIR": METRICS_DIR,
        "LOGS_DIR": LOGS_DIR,
        "FIGURES_DIR": FIGURES_DIR,
        "PREDICTIONS_DIR": PREDICTIONS_DIR,
        "EXPERIMENTS_DIR": EXPERIMENTS_DIR,
        "MLFLOW_TRACKING_DIR": MLFLOW_TRACKING_DIR,
        "MODEL_CHECKPOINT_PATH": model_ckpt_path,
        "LOSS_ACC_PLOT_PATH": loss_acc_plot_path,
    }

def clean_outputs_dir():
    if OUTPUTS_DIR.exists():
        print(f"[INFO] Cleaning outputs directory from previous experiment artifacts: {OUTPUTS_DIR}\n")
        for item in OUTPUTS_DIR.iterdir():
            if item.is_dir():
                shutil.rmtree(item)
            else:
                item.unlink()
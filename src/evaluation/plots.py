import matplotlib
matplotlib.use("Agg")  # prevent tkinter-related issues from saving plot - Agg is non-interactive backend. Doesn't require GUI. This headless environemnts are recomended for docker containers, environments without display etc.
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
import math
import torch

from src.training.engine import PyTorchTrainer

def loss_acc_plot(results):
    epochs = range(len(results['train_loss']))
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))

    ax[0].plot(epochs, results['train_loss'], label='Train')
    ax[0].plot(epochs, results['val_loss'], label='Validation')
    ax[0].set_title("Loss")
    ax[0].legend()

    ax[1].plot(epochs, results['train_acc'], label='Train')
    ax[1].plot(epochs, results['val_acc'], label='Validation')
    ax[1].set_title("Accuracy")
    ax[1].legend()
    
    return fig

def plot_roc_ovr(model_dict, test_dataloader, class_names):
    """
    Plots OvR ROC curves for each class, comparing all models.

    Args:
        model_dict: dict[arch] = {
            "model": torch.nn.Module,
            "run_id": str,
            "hu": int,
            "lr": float
        }
        test_dataloader: DataLoader
        class_names: list of class labels

    Returns:
        fig: matplotlib Figure
        auc_scores: dict[class_name][legend_label] = {"auc": float, "run_id": str}
    """
    # Get true labels from first model (all models use same test set)
    first_model = next(iter(model_dict.values()))["model"]
    
    # Create a temporary trainer just for predictions
    dummy_loss_fn = torch.nn.CrossEntropyLoss()
    dummy_optimizer = torch.optim.SGD(first_model.parameters(), lr=0.01)
    trainer = PyTorchTrainer(first_model, dummy_loss_fn, dummy_optimizer)
    
    y_true, _ = trainer.get_predictions(test_dataloader)
    y_bin = label_binarize(y_true, classes=list(range(len(class_names))))

    # Setup plot grid
    num_classes = len(class_names)
    cols = 3
    rows = math.ceil(num_classes / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows))
    axes = axes.flatten()

    auc_scores = {}

    for idx, class_name in enumerate(class_names):
        ax = axes[idx]
        auc_scores[class_name] = {}

        # Loop through each model
        for arch_name, info in model_dict.items():
            model = info["model"]
            run_id = info["run_id"]
            hu = info["hu"]
            lr = info["lr"]
            
            legend_label = f"{arch_name}_hu{hu}_lr{lr}"
            
            # Create trainer for this model
            trainer = PyTorchTrainer(
                model, 
                torch.nn.CrossEntropyLoss(), 
                torch.optim.SGD(model.parameters(), lr=0.01)
            )
            
            # Evaluate this specific model
            _, y_probs = trainer.get_predictions(test_dataloader)
            
            # Compute ROC curve for this class
            fpr, tpr, _ = roc_curve(y_bin[:, idx], y_probs[:, idx])
            roc_auc = auc(fpr, tpr)
            
            # Plot the curve
            ax.plot(fpr, tpr, label=f"{legend_label} (AUC = {roc_auc:.3f})")
            
            # Store AUC score
            auc_scores[class_name][legend_label] = {
                "auc": roc_auc,
                "run_id": run_id
            }

        ax.plot([0, 1], [0, 1], 'k--')
        ax.set_title(f"ROC - {class_name}")
        ax.set_xlabel("FPR")
        ax.set_ylabel("TPR")
        ax.grid(True)
        ax.legend()

    for i in range(len(class_names), len(axes)):
        fig.delaxes(axes[i])

    fig.suptitle("ROC Curves Per Class (OvR)", fontsize=16)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    return fig, auc_scores


def plot_roc_ovo(model_dict, test_dataloader, class_names):
    pass
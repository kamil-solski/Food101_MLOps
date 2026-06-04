import mlflow
import mlflow.pytorch
from mlflow.models.signature import infer_signature
import torch
import gc
import time
from tqdm import tqdm

from src.training.engine import PyTorchTrainer

# Required external components: your data loading and models
def train_with_mlflow(model: torch.nn.Module,
                      model_name: str,
                      train_dataloader,
                      val_dataloader,
                      loss_fn,
                      optimizer,
                      learning_rate,
                      epochs,
                      paths: dict,
                      device: str = None):
    """
    Train PyTorch model with MLflow tracking using the PyTorchTrainer.
    
    Args:
        model: PyTorch model to train
        model_name: Name of the model for logging
        train_dataloader: Training data loader
        val_dataloader: Validation data loader
        loss_fn: Loss function
        optimizer: Optimizer
        learning_rate: Learning rate
        epochs: Number of training epochs
        paths: Dictionary of paths for saving outputs
        device: Device to use ('cuda' or 'cpu', auto-detected if None)
    
    Returns:
        dict: Training results with metrics history
    """
    start_time = time.time()
    
    # Initialize trainer
    trainer = PyTorchTrainer(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        device=device
    )
    
    # Log parameters
    mlflow.log_param("model", model_name)
    mlflow.log_param("learning_rate", learning_rate)
    mlflow.log_param("epochs", epochs)
    mlflow.log_param("batch_size", train_dataloader.batch_size)
    mlflow.log_param("optimizer", type(optimizer).__name__)
    mlflow.log_param("loss_fn", type(loss_fn).__name__)
    mlflow.log_param("device", trainer.device)
    
    results = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": []
    }
    
    progress = tqdm(range(epochs),
                    desc=f"Training {model_name}",
                    dynamic_ncols=True,
                    leave=False)
    
    for epoch in progress:
        # Training step
        train_loss, train_acc = trainer.train_step(train_dataloader)
        
        # Validation step
        val_loss, val_acc = trainer.eval_step(val_dataloader)
        
        progress.set_postfix({
            "train_loss": f"{train_loss:.4f}",
            "train_acc": f"{train_acc*100:.2f}%",
            "val_loss": f"{val_loss:.4f}",
            "val_acc": f"{val_acc*100:.2f}%"
        })

        # Log metrics to MLFlow
        mlflow.log_metric("train_loss", train_loss, step=epoch)
        mlflow.log_metric("train_acc", train_acc, step=epoch)
        mlflow.log_metric("val_loss", val_loss, step=epoch)
        mlflow.log_metric("val_acc", val_acc, step=epoch)

        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["val_loss"].append(val_loss)
        results["val_acc"].append(val_acc)

    # Save model
    total_time = time.time() - start_time
    mlflow.log_metric("train_time_sec", total_time)
    mlflow.set_tag("training_time_readable", f"{total_time:.2f} sec")
    
    model_path = paths["MODEL_CHECKPOINT_PATH"]  # raw models. They could be removed in the future. Mlflow doesn't use those checkpoints anyway
    torch.save(model.state_dict(), model_path)
        
    device = trainer.device
    
    # Prepare input example and signature
    example_input, _ = next(iter(val_dataloader))
    example_input = example_input.to(device)
    example_output = model(example_input)
    signature = infer_signature(example_input.cpu().numpy(), example_output.detach().cpu().numpy())
    
    mlflow.pytorch.log_model(
        model, 
        name="model",
        input_example=example_input[:1].cpu().numpy(),
        signature=signature
    )

    return results
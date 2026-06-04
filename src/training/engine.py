import torch
import numpy as np
from typing import Tuple
from abc import ABC, abstractmethod


class BaseTrainer(ABC):
    """Abstract base class for all trainers."""
    
    @abstractmethod
    def train_step(self, *args, **kwargs):
        """Single training step/epoch."""
        pass
    
    @abstractmethod
    def eval_step(self, *args, **kwargs):
        """Evaluation step."""
        pass
    
    @abstractmethod
    def get_predictions(self, *args, **kwargs):
        """Get predictions for evaluation."""
        pass


class PyTorchTrainer(BaseTrainer):
    """Trainer for PyTorch neural networks."""
    
    def __init__(self, model: torch.nn.Module, loss_fn: torch.nn.Module,
                 optimizer: torch.optim.Optimizer, device: str = None):
        """
        Args:
            model: PyTorch model
            loss_fn: Loss function
            optimizer: Optimizer
            device: 'cuda' or 'cpu' (auto-detected if None)
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.device = device
        self.model = model.to(self.device)
        self.loss_fn = loss_fn
        self.optimizer = optimizer
    
    def train_step(self, dataloader: torch.utils.data.DataLoader) -> Tuple[float, float]:
        """
        Single training epoch.
        
        Args:
            dataloader: Training data loader
            
        Returns:
            Tuple of (train_loss, train_acc)
        """
        self.model.train()
        train_loss, train_acc = 0, 0
        
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(self.device), y.to(self.device)
            
            # Forward pass
            y_pred = self.model(X)
            loss = self.loss_fn(y_pred, y)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Accumulate metrics
            train_loss += loss.item()
            y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
            train_acc += (y_pred_class == y).sum().item() / len(y_pred)
        
        train_loss /= len(dataloader)
        train_acc /= len(dataloader)
        
        return train_loss, train_acc
    
    def eval_step(self, dataloader: torch.utils.data.DataLoader) -> Tuple[float, float]:
        """
        Validation/test epoch.
        
        Args:
            dataloader: Validation/test data loader
            
        Returns:
            Tuple of (test_loss, test_acc)
        """
        self.model.eval()
        test_loss, test_acc = 0, 0
        
        with torch.inference_mode():
            for X, y in dataloader:
                X, y = X.to(self.device), y.to(self.device)
                
                # Forward pass
                test_pred_logits = self.model(X)
                loss = self.loss_fn(test_pred_logits, y)
                
                # Accumulate metrics
                test_loss += loss.item()
                test_pred_labels = test_pred_logits.argmax(dim=1)
                test_acc += ((test_pred_labels == y).sum().item() / len(test_pred_labels))
        
        test_loss /= len(dataloader)
        test_acc /= len(dataloader)
        
        return test_loss, test_acc
    
    def get_predictions(self, dataloader: torch.utils.data.DataLoader) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get predictions for evaluation.
        
        Args:
            dataloader: Data loader for predictions
            
        Returns:
            Tuple of (y_true, y_probs) as numpy arrays
        """
        self.model.eval()
        probs_list = []
        labels_list = []
        
        with torch.inference_mode():
            for X_batch, y_batch in dataloader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                
                logits = self.model(X_batch)
                probs = torch.softmax(logits, dim=1)
                probs_list.append(probs.cpu())
                labels_list.append(y_batch.cpu())
        
        y_true = torch.cat(labels_list).numpy()
        y_probs = torch.cat(probs_list).numpy()
        
        return y_true, y_probs
    
    def get_model(self) -> torch.nn.Module:
        """Return trained model."""
        return self.model


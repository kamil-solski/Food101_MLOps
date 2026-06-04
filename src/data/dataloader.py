from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os

from src.utils.paths import get_paths


def get_dataloaders(image_size: int, batch_size: int, fold=None, mode="train"):
    """
    Get PyTorch DataLoaders for training, validation, or testing.
    
    Args:
        image_size: Size to resize images to
        batch_size: Batch size for DataLoader
        fold: Can be:
            - Path object (e.g., Path("Data/Food101/fold_1")) → cross-validation mode
            - str (e.g., "fold_1") → cross-validation mode  
            - None → simple train/val/test structure (use mode to distinguish)
        mode: One of:
            - "train" → return (train_loader, val_loader, class_names)
            - "test" → return (test_loader, class_names)
    
    Returns:
        Training mode: (train_loader, val_loader, class_names)
        Test mode: (test_loader, class_names)
    """
    paths = get_paths(fold=fold)
    
    train_dir = paths["TRAIN_DIR"]
    val_dir = paths["VAL_DIR"]
    test_dir = paths["TEST_DIR"]  # remember this is in dataset_dir not in each folder
    classes_file = paths["CLASSES_FILE"]

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor()
    ])

    # workers setting
    nw = max(0, (os.cpu_count() or 0) // 4)
    pw = nw > 0

    # Load class names manually from file (always needed)
    with open(classes_file, "r") as f:
        class_names = [line.strip() for line in f.readlines()]

    # Load datasets based on mode parameter
    if mode == "train":  # Training phase: return train_loader, val_loader, class_names
        train_data = datasets.ImageFolder(train_dir, transform=transform)
        train_loader = DataLoader(
            train_data, 
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=nw,
            persistent_workers=pw
        )  

        val_data = datasets.ImageFolder(val_dir, transform=transform)
        val_loader = DataLoader(
            val_data, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=nw, 
            persistent_workers=pw
        )
        
        return train_loader, val_loader, class_names
        
    elif mode == "test":  # Evaluation phase: return test_loader, class_names
        test_data = datasets.ImageFolder(test_dir, transform=transform)
        test_loader = DataLoader(
            test_data, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=nw, 
            persistent_workers=pw
        )
        
        return test_loader, class_names
    
    else:
        raise ValueError(f"Invalid mode '{mode}'. Expected 'train' or 'test'.")
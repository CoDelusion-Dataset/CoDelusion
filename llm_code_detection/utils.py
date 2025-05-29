"""
Utility functions for code classification.

This module provides evaluation functions and other utilities.
"""

import os
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import precision_score, recall_score, f1_score
from torch.utils.data import DataLoader

from config import logger, DEVICE


def set_seed(seed: int) -> None:
    """
    Set random seed for reproducibility
    
    Args:
        seed: Random seed value
    """
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        
    logger.info(f"Random seed set to {seed}")


def evaluate(model: nn.Module, dataloader: DataLoader, device: torch.device = DEVICE) -> Dict:
    """
    Evaluate model performance
    
    Args:
        model: Trained model
        dataloader: Data loader
        device: Computation device
        
    Returns:
        Dictionary containing evaluation metrics
    """
    model.eval()
    val_preds = []
    val_labels = []
    val_loss = 0
    loss_fn = torch.nn.CrossEntropyLoss() 
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            # Calculate validation loss
            loss = loss_fn(outputs, labels)
            val_loss += loss.item()
            
            preds = torch.argmax(outputs, dim=1)
            val_preds.extend(preds.cpu().numpy())
            val_labels.extend(labels.cpu().numpy())
    
    precision = precision_score(val_labels, val_preds)
    recall = recall_score(val_labels, val_preds)
    f1 = f1_score(val_labels, val_preds)
    acc = np.mean(np.array(val_preds) == np.array(val_labels))
    
    return {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'loss': val_loss / len(dataloader)
    }


def plot_loss_curve(train_losses: List[float], val_losses: List[float], 
                   epochs: int, output_dir: str) -> str:
    """
    Plot training and validation loss curves
    
    Args:
        train_losses: List of training losses
        val_losses: List of validation losses
        epochs: Number of training epochs
        output_dir: Directory to save the plot
        
    Returns:
        Path to the saved plot file
    """
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, epochs + 1), train_losses, label='Training Loss')
    plt.plot(range(1, epochs + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f'loss_curve.png')
    plt.savefig(output_path)
    plt.close()
    
    return output_path


def save_model(model: nn.Module, output_dir: str, filename: str = 'best_model.pth') -> str:
    """
    Save model weights to file
    
    Args:
        model: Model to save
        output_dir: Directory to save the model
        filename: Filename to save the model
        
    Returns:
        Path to the saved model file
    """
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)
    torch.save(model.state_dict(), output_path)
    return output_path 
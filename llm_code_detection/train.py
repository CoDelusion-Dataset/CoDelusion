"""
Training module for code classification.

This module provides training functionality for CodeBERT classifier.
"""

import os
from typing import Dict, Tuple

import torch
from torch.nn import CrossEntropyLoss
from transformers import AdamW, get_linear_schedule_with_warmup

from config import (
    logger, DEVICE, DEFAULT_EPOCHS, DEFAULT_LEARNING_RATE, 
    DEFAULT_MAX_GRAD_NORM, DEFAULT_WARMUP_STEPS, DEFAULT_OUTPUT_DIR
)
from model import CodeBERTClassifier
from utils import evaluate, plot_loss_curve, save_model


def train_model(
    model: CodeBERTClassifier,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    epochs: int = DEFAULT_EPOCHS, 
    learning_rate: float = DEFAULT_LEARNING_RATE,
    warmup_steps: int = DEFAULT_WARMUP_STEPS,
    max_grad_norm: float = DEFAULT_MAX_GRAD_NORM,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    save_best: bool = True,
    plot_loss: bool = True,
    device: torch.device = DEVICE
) -> Tuple[Dict, str, str]:
    """
    Train the model
    
    Args:
        model: Model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        epochs: Number of training epochs
        learning_rate: Learning rate
        warmup_steps: Number of warmup steps
        max_grad_norm: Maximum gradient norm for clipping
        output_dir: Directory to save the model and plots
        save_best: Whether to save the best model
        plot_loss: Whether to plot loss curves
        device: Device to train on
        
    Returns:
        best_metrics: Dictionary of best metrics
        model_path: Path to the saved model (or None if not saved)
        plot_path: Path to the loss plot (or None if not plotted)
    """
    model.to(device)
    
    # Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    # Loss function
    loss_fn = CrossEntropyLoss()
    
    # Training state tracking
    train_losses = []
    val_losses = []
    best_f1 = 0
    best_metrics = None
    model_path = None
    plot_path = None
    
    logger.info("Starting training...")
    logger.info(f"Training device: {device}")
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            # Calculate loss
            loss = loss_fn(outputs, labels)
            total_loss += loss.item()
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            
            # Update weights
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        
        # Validation
        val_metrics = evaluate(model, val_loader, device)
        avg_loss = total_loss / len(train_loader)
        train_losses.append(avg_loss)
        val_losses.append(val_metrics['loss'])

        # Log metrics
        logger.info(f'Epoch {epoch+1}/{epochs}')
        logger.info(f'Training Loss: {avg_loss:.4f}')
        logger.info(f"Validation Accuracy: {val_metrics['accuracy']:.4f}")
        logger.info(f"Validation Precision: {val_metrics['precision']:.4f}")
        logger.info(f"Validation Recall: {val_metrics['recall']:.4f}")
        logger.info(f"Validation F1: {val_metrics['f1']:.4f}")
        logger.info('-' * 50)
        
        # Save best model
        if val_metrics['f1'] > best_f1:
            best_f1 = val_metrics['f1']
            best_metrics = val_metrics.copy()
            
            if save_best:
                model_path = save_model(model, output_dir)
                logger.info(f"Best model saved to {model_path}, F1: {best_f1:.4f}")
    
    logger.info(f'Training complete. Best validation F1: {best_f1:.4f}')

    # Plot loss curves
    if plot_loss:
        plot_path = plot_loss_curve(train_losses, val_losses, epochs, output_dir)
        logger.info(f"Loss curve saved to {plot_path}")
    
    return best_metrics, model_path, plot_path 
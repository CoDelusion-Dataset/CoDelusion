#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CodeBERT-based code classification main entry point.

This script serves as the entry point for training and evaluating the model.
"""

import argparse

from transformers import AutoTokenizer

from config import (
    logger, DEFAULT_SEED, DEFAULT_MAX_LEN, DEFAULT_BATCH_SIZE, 
    DEFAULT_EPOCHS, DEFAULT_LEARNING_RATE, DEFAULT_TEST_SIZE,
    DEFAULT_MODEL_PATH, DEFAULT_NUM_WORKERS, DEFAULT_OUTPUT_DIR,
    DEFAULT_WARMUP_STEPS, DEFAULT_MAX_GRAD_NORM, DEVICE
)
from data import load_and_preprocess_data, create_data_loaders
from model import CodeBERTClassifier
from train import train_model
from utils import set_seed


def parse_args():
    """
    Parse command line arguments
    
    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(description='CodeBERT-based code classifier')
    
    # Data parameters
    parser.add_argument('--data_path', type=str, default='./output_utf8.csv',
                        help='Path to the data file')
    parser.add_argument('--balance_data', action='store_true',
                        help='Whether to balance positive and negative samples')
    parser.add_argument('--test_size', type=float, default=DEFAULT_TEST_SIZE,
                        help='Test set proportion')
    
    # Model parameters
    parser.add_argument('--model_path', type=str, default=DEFAULT_MODEL_PATH,
                        help='Path to the CodeBERT model')
    parser.add_argument('--freeze_base_model', action='store_true',
                        help='Whether to freeze CodeBERT base model parameters')
    
    # Training parameters
    parser.add_argument('--max_len', type=int, default=DEFAULT_MAX_LEN,
                        help='Maximum sequence length')
    parser.add_argument('--batch_size', type=int, default=DEFAULT_BATCH_SIZE,
                        help='Batch size')
    parser.add_argument('--epochs', type=int, default=DEFAULT_EPOCHS,
                        help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=DEFAULT_LEARNING_RATE,
                        help='Learning rate')
    parser.add_argument('--warmup_steps', type=int, default=DEFAULT_WARMUP_STEPS,
                        help='Number of warmup steps')
    parser.add_argument('--max_grad_norm', type=float, default=DEFAULT_MAX_GRAD_NORM,
                        help='Gradient clipping threshold')
    parser.add_argument('--num_workers', type=int, default=DEFAULT_NUM_WORKERS,
                        help='Number of data loader workers')
    parser.add_argument('--seed', type=int, default=DEFAULT_SEED,
                        help='Random seed')
    
    # Output parameters
    parser.add_argument('--output_dir', type=str, default=DEFAULT_OUTPUT_DIR,
                        help='Output directory')
    parser.add_argument('--save_model', action='store_true',
                        help='Whether to save the model')
    parser.add_argument('--plot_loss', action='store_true',
                        help='Whether to plot the loss curve')
    
    return parser.parse_args()


def main():
    """
    Main execution function
    """
    # Parse command line arguments
    args = parse_args()
    
    # Set random seed for reproducibility
    set_seed(args.seed)
    
    # Load and preprocess data
    logger.info(f"Loading data from {args.data_path}...")
    df = load_and_preprocess_data(
        data_path=args.data_path,
        seed=args.seed,
        balance=args.balance_data
    )
    
    # Initialize tokenizer
    logger.info(f"Loading tokenizer from {args.model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    
    # Create data loaders
    train_loader, val_loader, train_size, val_size = create_data_loaders(
        df=df,
        tokenizer=tokenizer,
        max_len=args.max_len,
        batch_size=args.batch_size,
        test_size=args.test_size,
        seed=args.seed,
        num_workers=args.num_workers
    )
    
    # Initialize model
    logger.info("Initializing model...")
    model = CodeBERTClassifier(
        model_path=args.model_path,
        freeze_base=args.freeze_base_model
    )
    
    # Train model
    best_metrics, model_path, plot_path = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        max_grad_norm=args.max_grad_norm,
        output_dir=args.output_dir,
        save_best=args.save_model,
        plot_loss=args.plot_loss,
        device=DEVICE
    )
    
    # Print final results
    logger.info("Training completed!")
    logger.info(f"Best metrics: Accuracy={best_metrics['accuracy']:.4f}, "
                f"Precision={best_metrics['precision']:.4f}, "
                f"Recall={best_metrics['recall']:.4f}, "
                f"F1={best_metrics['f1']:.4f}")
    
    if model_path:
        logger.info(f"Best model saved to: {model_path}")
    
    if plot_path:
        logger.info(f"Loss curve saved to: {plot_path}")


if __name__ == '__main__':
    main() 
"""
Data handling module for code classification.

This module provides dataset classes and data processing functions.
"""

from typing import Dict, List, Tuple

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

from config import logger, DEFAULT_TEST_SIZE, DEFAULT_SEED, DEFAULT_BATCH_SIZE, DEFAULT_NUM_WORKERS


class CodeDataset(Dataset):
    """
    Code dataset class for processing code and docstring pairs
    """
    def __init__(self, descriptions: List[str], codes: List[str], 
                labels: List[int], tokenizer, max_len: int):
        """
        Initialize the dataset
        
        Args:
            descriptions: List of code descriptions/docstrings
            codes: List of code snippets
            labels: List of labels (0 or 1)
            tokenizer: Tokenizer for processing text
            max_len: Maximum sequence length
        """
        self.descriptions = descriptions
        self.codes = codes
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
        
    def __len__(self) -> int:
        return len(self.labels)
    
    def __getitem__(self, idx: int) -> Dict:
        description = str(self.descriptions[idx])
        code = str(self.codes[idx])
        
        # Concatenate docstring and code as input
        combined = f"[CLS]{description}[SEP]{code}[SEP]"
        encoding = self.tokenizer.encode_plus(
            combined,
            add_special_tokens=False,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }


def load_and_preprocess_data(data_path: str, seed: int = DEFAULT_SEED, 
                            balance: bool = True) -> pd.DataFrame:
    """
    Load and preprocess data
    
    Args:
        data_path: Path to the data file
        seed: Random seed
        balance: Whether to balance positive and negative samples
        
    Returns:
        Processed DataFrame
    """
    df = pd.read_csv(data_path)
    
    # Count positive and negative samples
    pos_num = df[df['label'] == 1].shape[0]
    neg_num = df[df['label'] == 0].shape[0]
    logger.info(f"Original dataset: Positive samples: {pos_num}, Negative samples: {neg_num}")
    
    # Balance the dataset if needed
    if balance and pos_num != neg_num:
        if pos_num > neg_num:
            pos_df = df[df['label'] == 1].sample(neg_num, random_state=seed)
            neg_df = df[df['label'] == 0]
        else:
            pos_df = df[df['label'] == 1]
            neg_df = df[df['label'] == 0].sample(pos_num, random_state=seed)
            
        df = pd.concat([pos_df, neg_df])
        pos_num = df[df['label'] == 1].shape[0]
        neg_num = df[df['label'] == 0].shape[0]
        logger.info(f"Balanced dataset: Positive samples: {pos_num}, Negative samples: {neg_num}")
    
    return df


def create_data_loaders(df: pd.DataFrame, tokenizer, max_len: int, batch_size: int = DEFAULT_BATCH_SIZE,
                       test_size: float = DEFAULT_TEST_SIZE, seed: int = DEFAULT_SEED,
                       num_workers: int = DEFAULT_NUM_WORKERS) -> Tuple[DataLoader, DataLoader, int, int]:
    """
    Create train and validation data loaders
    
    Args:
        df: DataFrame containing the data
        tokenizer: Tokenizer for processing text
        max_len: Maximum sequence length
        batch_size: Batch size
        test_size: Proportion of data to use for validation
        seed: Random seed
        num_workers: Number of workers for data loading
        
    Returns:
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        train_size: Number of training samples
        val_size: Number of validation samples
    """
    # Split dataset
    train_df, val_df = train_test_split(
        df,
        test_size=test_size,
        random_state=seed,
        stratify=df['label']
    )
    
    logger.info(f"Training set: {len(train_df)} samples, Validation set: {len(val_df)} samples")
    
    # Create datasets
    train_dataset = CodeDataset(
        descriptions=train_df.docstring.values,
        codes=train_df.code.values,
        labels=train_df.label.values,
        tokenizer=tokenizer,
        max_len=max_len
    )
    
    val_dataset = CodeDataset(
        descriptions=val_df.docstring.values,
        codes=val_df.code.values,
        labels=val_df.label.values,
        tokenizer=tokenizer,
        max_len=max_len
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    return train_loader, val_loader, len(train_df), len(val_df) 
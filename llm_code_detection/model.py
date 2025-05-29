"""
Model definition for code classification.

This module provides the CodeBERT-based classifier model.
"""

import torch
import torch.nn as nn
from transformers import AutoModel

from config import DEFAULT_MODEL_PATH


class CodeBERTClassifier(nn.Module):
    """
    CodeBERT-based code classifier
    """
    def __init__(self, model_path: str = DEFAULT_MODEL_PATH, freeze_base: bool = False):
        """
        Initialize the model
        
        Args:
            model_path: Path to the CodeBERT model
            freeze_base: Whether to freeze the CodeBERT base model parameters
        """
        super(CodeBERTClassifier, self).__init__()
        self.codebert = AutoModel.from_pretrained(model_path)
        self.dropout = nn.Dropout(0.1)
        
        # Get hidden layer dimension (matches [CLS] vector dimension)
        hidden_size = self.codebert.config.hidden_size
        
        # Set whether to freeze base model parameters
        for param in self.codebert.parameters():
            param.requires_grad = not freeze_base
            
        # Classification layer
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 2)  # Binary classification output
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            
        Returns:
            logits: Classification logits
        """
        # Get full hidden states [batch_size, seq_len, hidden_size]
        outputs = self.codebert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        # Extract [CLS] token's hidden state (first token)
        cls_output = outputs.last_hidden_state[:, 0, :]  # [batch_size, hidden_size]
        
        # Classification processing
        cls_output = self.dropout(cls_output)
        logits = self.classifier(cls_output)  # [batch_size, 2]
        
        return logits 
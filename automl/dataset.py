"""
Shared Dataset Module
Provides a single TextDataset used by both model_trainer and evaluator.
"""

import torch
from torch.utils.data import Dataset
from typing import List


class TextDataset(Dataset):
    """PyTorch Dataset for text classification.

    Pre-tokenizes all texts once at construction time and stores the resulting
    tensors.  This eliminates redundant tokenizer calls that would otherwise
    fire on every __getitem__ access — i.e. every sample × every epoch ×
    every Optuna proxy trial and evaluation pass.
    """

    def __init__(self, texts: List[str], labels, tokenizer, max_length: int):
        # Batch-tokenize the full list once; results are plain tensors stored in
        # self.encodings.  Each __getitem__ just indexes into them.
        self.encodings = tokenizer(
            [str(t) for t in texts],
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )
        # Handle both single-label (1-D int) and multi-label (2-D float) arrays.
        import numpy as np
        labels_arr = np.asarray(labels)
        if labels_arr.ndim == 2:
            # Multi-label: multi-hot float matrix of shape (N, num_classes)
            self.labels = torch.tensor(labels_arr, dtype=torch.float)
        else:
            # Single-label: integer class IDs
            self.labels = torch.tensor(list(labels), dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'input_ids':      self.encodings['input_ids'][idx],
            'attention_mask': self.encodings['attention_mask'][idx],
            'labels':         self.labels[idx],
        }

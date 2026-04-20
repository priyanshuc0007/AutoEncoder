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

    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_length: int):
        # Batch-tokenize the full list once; results are plain tensors stored in
        # self.encodings.  Each __getitem__ just indexes into them.
        self.encodings = tokenizer(
            [str(t) for t in texts],
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )
        self.labels = torch.tensor(list(labels), dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'input_ids':      self.encodings['input_ids'][idx],
            'attention_mask': self.encodings['attention_mask'][idx],
            'labels':         self.labels[idx],
        }

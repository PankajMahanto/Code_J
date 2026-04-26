"""
============================================================================
 src/data/dataset.py
 ---------------------------------------------------------------------------
 PyTorch dataset + dataloader helpers.
============================================================================
"""
from __future__ import annotations

import torch
from torch.utils.data import Dataset, DataLoader
from scipy import sparse


class BoWDataset(Dataset):
    """
    Yields (x_norm, x_raw) per document:
      • x_norm — L1-normalised BoW (input to encoder)
      • x_raw  — raw counts (target for reconstruction loss)
    """

    def __init__(self, bow_sparse: sparse.csr_matrix):
        dense = torch.tensor(bow_sparse.toarray(), dtype=torch.float32)
        self.x_raw  = dense
        self.x_norm = dense / dense.sum(dim=1, keepdim=True).clamp(min=1.0)

    def __len__(self) -> int:
        return self.x_raw.size(0)

    def __getitem__(self, idx: int):
        return self.x_norm[idx], self.x_raw[idx]


def make_dataloader(bow_sparse: sparse.csr_matrix,
                    batch_size: int,
                    shuffle: bool = True,
                    num_workers: int = 0) -> DataLoader:
    ds = BoWDataset(bow_sparse)
    return DataLoader(ds,
                      batch_size=batch_size,
                      shuffle=shuffle,
                      drop_last=True,
                      num_workers=num_workers)

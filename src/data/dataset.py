"""
============================================================================
 src/data/dataset.py
 ---------------------------------------------------------------------------
 PyTorch dataset + dataloader helpers.

 Each sample yields a triple:
   • x_norm — L1-normalised BoW (kept for backwards compatibility)
   • x_raw  — raw counts (target for reconstruction loss + encoder input
              after a log1p in the trainer)
   • x_ctx  — sentence-transformer document embedding (zeros if absent)
============================================================================
"""
from __future__ import annotations

import torch
from torch.utils.data import Dataset, DataLoader
from scipy import sparse


class BoWDataset(Dataset):
    def __init__(self,
                 bow_sparse: sparse.csr_matrix,
                 ctx_embeds: torch.Tensor | None = None):
        dense = torch.tensor(bow_sparse.toarray(), dtype=torch.float32)
        self.x_raw  = dense
        self.x_norm = dense / dense.sum(dim=1, keepdim=True).clamp(min=1.0)

        if ctx_embeds is None:
            self.x_ctx = torch.zeros(dense.size(0), 0, dtype=torch.float32)
        else:
            if ctx_embeds.size(0) != dense.size(0):
                raise ValueError(
                    f"ctx_embeds size {ctx_embeds.size(0)} does not match "
                    f"bow rows {dense.size(0)}"
                )
            self.x_ctx = ctx_embeds.float()

    def __len__(self) -> int:
        return self.x_raw.size(0)

    def __getitem__(self, idx: int):
        return self.x_norm[idx], self.x_raw[idx], self.x_ctx[idx]


def make_dataloader(bow_sparse: sparse.csr_matrix,
                    batch_size: int,
                    ctx_embeds: torch.Tensor | None = None,
                    shuffle: bool = True,
                    num_workers: int = 0) -> DataLoader:
    ds = BoWDataset(bow_sparse, ctx_embeds=ctx_embeds)
    return DataLoader(ds,
                      batch_size=batch_size,
                      shuffle=shuffle,
                      drop_last=True,
                      num_workers=num_workers)

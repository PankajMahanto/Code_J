"""
============================================================================
 src/data/dataset.py  [REWRITE]
 ---------------------------------------------------------------------------
 PyTorch dataset + dataloader helpers for the Contextual Topic Model
 architecture.

 Each item now yields THREE tensors:
   • x_norm — L1-normalised BoW (legacy, kept for compatibility)
   • x_raw  — raw word counts (target for the multinomial reconstruction)
   • x_ctx  — dense contextual document embedding from a sentence-
              transformer; this gets concatenated with the BoW input
              inside the encoder before being projected to the latent
              topic space.
============================================================================
"""
from __future__ import annotations

from typing import Optional

import torch
from torch.utils.data import Dataset, DataLoader
from scipy import sparse


class BoWDataset(Dataset):
    """
    Yields (x_norm, x_raw, x_ctx) per document.
    """

    def __init__(self,
                 bow_sparse: sparse.csr_matrix,
                 ctx_embeds: Optional[torch.Tensor] = None):
        dense = torch.tensor(bow_sparse.toarray(), dtype=torch.float32)
        self.x_raw  = dense
        self.x_norm = dense / dense.sum(dim=1, keepdim=True).clamp(min=1.0)

        if ctx_embeds is None:
            # Fallback: zeros (not recommended — kept for unit tests).
            self.x_ctx = torch.zeros(dense.size(0), 1, dtype=torch.float32)
        else:
            assert ctx_embeds.size(0) == dense.size(0), (
                f"ctx_embeds rows ({ctx_embeds.size(0)}) must match "
                f"#docs ({dense.size(0)})"
            )
            self.x_ctx = ctx_embeds.to(torch.float32)

    def __len__(self) -> int:
        return self.x_raw.size(0)

    def __getitem__(self, idx: int):
        return self.x_norm[idx], self.x_raw[idx], self.x_ctx[idx]


def make_dataloader(bow_sparse: sparse.csr_matrix,
                    batch_size: int,
                    ctx_embeds: Optional[torch.Tensor] = None,
                    shuffle: bool = True,
                    num_workers: int = 0) -> DataLoader:
    ds = BoWDataset(bow_sparse, ctx_embeds=ctx_embeds)
    return DataLoader(ds,
                      batch_size=batch_size,
                      shuffle=shuffle,
                      drop_last=True,
                      num_workers=num_workers)

"""
============================================================================
 src/data/dataset.py  [REWRITE - v4]
 ---------------------------------------------------------------------------
 PyTorch dataset + dataloader helpers.

 v4 CHANGES
 ----------
 * Each item now yields (x_norm, x_raw, x_ctx) where x_ctx is the
   contextual document embedding produced by a sentence-transformer.
 * If doc embeddings are not supplied (e.g. legacy flow), x_ctx is a
   zero-length tensor so downstream code can branch on .numel().
============================================================================
"""
from __future__ import annotations

from typing import Optional

import torch
from torch.utils.data import Dataset, DataLoader
from scipy import sparse


class BoWDataset(Dataset):
    """
    Yields (x_norm, x_raw, x_ctx) per document:
      * x_norm - L1-normalised BoW (auxiliary view of the BoW)
      * x_raw  - raw counts (target for reconstruction loss)
      * x_ctx  - contextual document embedding (sentence-transformer);
                 shape [ctx_dim] or [0] when embeddings were not supplied.
    """

    def __init__(self,
                 bow_sparse: sparse.csr_matrix,
                 ctx_embeds: Optional[torch.Tensor] = None):
        dense = torch.tensor(bow_sparse.toarray(), dtype=torch.float32)
        self.x_raw  = dense
        self.x_norm = dense / dense.sum(dim=1, keepdim=True).clamp(min=1.0)

        if ctx_embeds is None:
            self.x_ctx = torch.zeros((dense.size(0), 0), dtype=torch.float32)
        else:
            if ctx_embeds.size(0) != dense.size(0):
                raise ValueError(
                    f"ctx_embeds rows ({ctx_embeds.size(0)}) != BoW rows "
                    f"({dense.size(0)}). Re-run preprocessing so they match."
                )
            self.x_ctx = ctx_embeds.to(dtype=torch.float32)

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

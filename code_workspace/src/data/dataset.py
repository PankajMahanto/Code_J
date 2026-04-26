"""
============================================================================
 src/data/dataset.py
 ---------------------------------------------------------------------------
 PyTorch dataset + dataloader helpers.

 Supports the contextual-topic-model upgrade: every document carries both a
 sparse BoW vector AND a dense sentence-transformer embedding.  These are
 concatenated inside the encoder before the first dense layer.
============================================================================
"""
from __future__ import annotations

from typing import Optional

import torch
from torch.utils.data import Dataset, DataLoader
from scipy import sparse


class BoWDataset(Dataset):
    """
    Yields a 3-tuple per document:

      • x_norm : L1-normalised BoW       (legacy compatibility)
      • x_raw  : raw integer BoW counts  (target for reconstruction loss)
      • x_ctx  : sentence-transformer embedding  (contextual signal)

    If ``ctx_embeds`` is ``None`` (e.g. ablation runs that disable the
    contextual head), an all-zero placeholder of shape [ctx_dim] is yielded
    so the rest of the pipeline doesn't have to special-case it.
    """

    def __init__(self,
                 bow_sparse: sparse.csr_matrix,
                 ctx_embeds: Optional[torch.Tensor] = None,
                 ctx_dim:    int = 0):
        dense = torch.tensor(bow_sparse.toarray(), dtype=torch.float32)
        self.x_raw  = dense
        self.x_norm = dense / dense.sum(dim=1, keepdim=True).clamp(min=1.0)

        if ctx_embeds is not None:
            assert ctx_embeds.shape[0] == dense.shape[0], (
                f"ctx_embeds rows ({ctx_embeds.shape[0]}) != docs "
                f"({dense.shape[0]})"
            )
            self.x_ctx = ctx_embeds.float()
        else:
            self.x_ctx = torch.zeros(dense.shape[0], ctx_dim, dtype=torch.float32)

    def __len__(self) -> int:
        return self.x_raw.size(0)

    def __getitem__(self, idx: int):
        return self.x_norm[idx], self.x_raw[idx], self.x_ctx[idx]


def make_dataloader(bow_sparse: sparse.csr_matrix,
                    batch_size: int,
                    shuffle: bool = True,
                    num_workers: int = 0,
                    ctx_embeds: Optional[torch.Tensor] = None,
                    ctx_dim:    int = 0) -> DataLoader:
    ds = BoWDataset(bow_sparse, ctx_embeds=ctx_embeds, ctx_dim=ctx_dim)
    return DataLoader(ds,
                      batch_size=batch_size,
                      shuffle=shuffle,
                      drop_last=True,
                      num_workers=num_workers)

"""
============================================================================
 src/data/dataset.py
 ---------------------------------------------------------------------------
 PyTorch dataset + dataloader helpers.

 The dataset now emits TWO views per document:
   • x_raw — raw integer counts  (target for reconstruction loss; also used
             as the BoW input after log1p normalisation inside the trainer).
   • x_ctx — dense contextual embedding from a sentence-transformer
             (e.g. `all-MiniLM-L6-v2`, 384-d).  Concatenated with the BoW
             inside the encoder.  Enables a Contextual Topic Model
             (Bianchi et al., 2021) in place of the deprecated GloVe path.
============================================================================
"""
from __future__ import annotations

from typing import Optional

import torch
from torch.utils.data import Dataset, DataLoader
from scipy import sparse


class BoWContextualDataset(Dataset):
    """Yields ``(x_raw, x_ctx)`` per document."""

    def __init__(self,
                 bow_sparse: sparse.csr_matrix,
                 context_embeds: Optional[torch.Tensor] = None):
        dense = torch.tensor(bow_sparse.toarray(), dtype=torch.float32)
        self.x_raw = dense

        if context_embeds is None:
            self.x_ctx = torch.zeros(dense.size(0), 0, dtype=torch.float32)
        else:
            if context_embeds.size(0) != dense.size(0):
                raise ValueError(
                    f"context_embeds rows ({context_embeds.size(0)}) "
                    f"do not match BoW rows ({dense.size(0)})."
                )
            self.x_ctx = context_embeds.float()

    def __len__(self) -> int:
        return self.x_raw.size(0)

    def __getitem__(self, idx: int):
        return self.x_raw[idx], self.x_ctx[idx]


def make_dataloader(bow_sparse: sparse.csr_matrix,
                    context_embeds: Optional[torch.Tensor] = None,
                    batch_size: int = 128,
                    shuffle: bool = True,
                    num_workers: int = 0) -> DataLoader:
    ds = BoWContextualDataset(bow_sparse, context_embeds)
    return DataLoader(ds,
                      batch_size=batch_size,
                      shuffle=shuffle,
                      drop_last=True,
                      num_workers=num_workers)


BoWDataset = BoWContextualDataset

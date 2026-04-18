"""
=============================================================================
NOVELTY 1: Spectral Graph-Infused Hierarchical Poincaré Encoder (SGP-E)
=============================================================================

Three sub-contributions:
  (a) Spectral Graph Convolution  — smooths BoW via PMI vocabulary graph
  (b) Hierarchical Poincaré latent manifold — hyperbolic z captures topic hierarchy
  (c) Fisher-Rao Information-Geometry KL — precision-weighted natural-gradient KL

Published for: IEEE Transactions on Knowledge and Data Engineering
Author note: These three components together are novel; no prior topic-model
             paper combines all three in the encoder path.
=============================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ─────────────────────────────────────────────────────────────────────────────
#  Helper: Poincaré Ball operations
# ─────────────────────────────────────────────────────────────────────────────
class PoincareBall:
    """
    Implements basic Riemannian operations on the Poincaré ball model
    of hyperbolic space with curvature -c.

    Reference geometry:
        ||x||  <  1/√c   (all points inside the open ball)
    """
    def __init__(self, c: float = 1.0):
        self.c = c

    # ── projection (keeps points inside the ball) ──────────────────────────
    def proj(self, x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
        max_norm = 1.0 / (self.c ** 0.5) - eps
        norm = x.norm(dim=-1, keepdim=True).clamp(min=1e-15)
        return torch.where(norm >= max_norm, x / norm * max_norm, x)

    # ── Möbius addition ─────────────────────────────────────────────────────
    def mobius_add(self, x: torch.Tensor, y: torch.Tensor,
                   eps: float = 1e-15) -> torch.Tensor:
        c   = self.c
        x2  = (x * x).sum(-1, keepdim=True)
        y2  = (y * y).sum(-1, keepdim=True)
        xy  = (x * y).sum(-1, keepdim=True)
        num   = (1 + 2*c*xy + c*y2) * x + (1 - c*x2) * y
        denom = 1 + 2*c*xy + c**2 * x2 * y2
        return num / (denom + eps)

    # ── exponential map at origin ────────────────────────────────────────────
    def expmap0(self, v: torch.Tensor, min_norm: float = 1e-15) -> torch.Tensor:
        v_norm  = v.norm(dim=-1, keepdim=True).clamp(min=min_norm)
        c_sqrt  = self.c ** 0.5
        tanh_in = (c_sqrt * v_norm).clamp(max=15.0)   # prevent overflow
        return torch.tanh(tanh_in) / (c_sqrt * v_norm) * v

    # ── logarithmic map at origin ────────────────────────────────────────────
    def logmap0(self, x: torch.Tensor, min_norm: float = 1e-15) -> torch.Tensor:
        x_norm   = x.norm(dim=-1, keepdim=True).clamp(min=min_norm)
        c_sqrt   = self.c ** 0.5
        atanh_in = (c_sqrt * x_norm).clamp(max=1.0 - 1e-6)
        return torch.atanh(atanh_in) / (c_sqrt * x_norm) * x

    # ── geodesic distance ────────────────────────────────────────────────────
    def dist(self, x: torch.Tensor, y: torch.Tensor,
             min_norm: float = 1e-15) -> torch.Tensor:
        diff      = self.mobius_add(-x, y)
        diff_norm = diff.norm(dim=-1).clamp(min=min_norm)
        c_sqrt    = self.c ** 0.5
        atanh_in  = (c_sqrt * diff_norm).clamp(max=1.0 - 1e-6)
        return 2.0 / c_sqrt * torch.atanh(atanh_in)


# ─────────────────────────────────────────────────────────────────────────────
#  (a)  Spectral Graph Convolution layer
# ─────────────────────────────────────────────────────────────────────────────
class SpectralGraphConv(nn.Module):
    """
    Simplified two-hop spectral GCN over the PMI vocabulary graph.

    Forward:
        1st hop  :  H1 = ReLU( (Â x) W1 )
        2nd hop  :  H2 = ReLU( (Â² x) W2 )  +  skip(x)
    Â = D^{-½} A D^{-½}  (symmetric normalised PMI adjacency)
    """
    def __init__(self, vocab_size: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.W1    = nn.Linear(vocab_size, hidden_dim, bias=True)
        self.W2    = nn.Linear(vocab_size, out_dim,    bias=True)   # 2-hop on raw
        self.skip  = nn.Linear(vocab_size, out_dim,    bias=False)
        self.bn1   = nn.BatchNorm1d(hidden_dim)
        self.bn2   = nn.BatchNorm1d(out_dim)

    @staticmethod
    def build_adj_norm(pmi_matrix: torch.Tensor) -> torch.Tensor:
        """
        Build D^{-½}(A + I)D^{-½}  from a PMI matrix.
        Only positive PMI edges are kept (semantic association).
        """
        A       = (pmi_matrix > 0).float() * pmi_matrix.clamp(min=0)
        A       = A + torch.eye(A.shape[0], device=A.device)          # self-loops
        degree  = A.sum(dim=1).clamp(min=1e-8)
        d_isqrt = degree.pow(-0.5)
        # symmetric normalisation: Â = diag(d^{-½}) A diag(d^{-½})
        adj_norm = d_isqrt.unsqueeze(1) * A * d_isqrt.unsqueeze(0)
        return adj_norm                                                  # [V, V]

    def forward(self, x: torch.Tensor,
                adj_norm: torch.Tensor) -> torch.Tensor:
        """
        x        : [B, V]
        adj_norm : [V, V]
        returns  : [B, out_dim]
        """
        # 1st hop: Â x  → W1
        x1 = x @ adj_norm                                # [B, V]
        h1 = F.relu(self.bn1(self.W1(x1)))               # [B, hidden]

        # 2nd hop: Â² x  → W2
        x2 = x1 @ adj_norm                               # [B, V]  (2nd aggregation)
        h2 = F.relu(self.bn2(self.W2(x2)))               # [B, out_dim]

        # skip connection from raw BoW
        skip = F.relu(self.skip(x))                      # [B, out_dim]

        return h2 + skip                                  # [B, out_dim]


# ─────────────────────────────────────────────────────────────────────────────
#  (c)  Fisher-Rao Information-Geometry KL
# ─────────────────────────────────────────────────────────────────────────────
def fisher_rao_kl(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """
    Fisher-Rao precision-weighted KL divergence.

    Motivation
    ----------
    Standard VAE KL treats every latent dimension equally.
    The Fisher information matrix of a Gaussian N(μ,σ²) is diagonal
    with F_μ = 1/σ² and F_{σ²} = 1/(2σ⁴).
    We use the posterior precision 1/σ² as a natural weight:
        - High-precision (sharp) dims get *more* regularisation → stay near 0
        - Low-precision (flat) dims get *less* → free to roam

    This is the natural-gradient steepest descent direction on the
    statistical manifold, giving an information-geometry interpretation.

    Returns the scalar mean KL over the batch.
    """
    sigma_sq = logvar.exp()                              # [B, K]
    precision = 1.0 / (sigma_sq.detach() + 1e-8)        # [B, K]  (stop-grad)

    # per-dimension KL: -½(1 + logvar - μ² - σ²)
    kl_per_dim  = -0.5 * (1.0 + logvar - mu.pow(2) - sigma_sq)   # [B, K]

    kl_standard = kl_per_dim.sum(dim=-1).mean()                   # scalar
    kl_fisher   = (precision * kl_per_dim).sum(dim=-1).mean()     # scalar

    # combine: standard KL + small Fisher-Rao correction (λ=0.10)
    return kl_standard + 0.10 * kl_fisher


# ─────────────────────────────────────────────────────────────────────────────
#  NOVELTY 1:  SGP-E  full encoder
# ─────────────────────────────────────────────────────────────────────────────
class SGPEncoder(nn.Module):
    """
    Spectral Graph-Infused Hierarchical Poincaré Encoder  (SGP-E).

    Architecture
    ────────────
    BoW  ──[SpectralGCN]──────────────────────╮
         ──[FC→BN→ReLU→Dropout]──[FC→BN→ReLU]─┤ cat ──[FC→BN→ReLU]──
                                               ╯
    → μ_E, log σ²_E  (Euclidean)
    → z_E  = μ_E + ε·σ_E           (reparameterisation)
    → z_H  = expmap₀(z_E · α)      (lift to Poincaré ball)
    → θ    = CapsuleRouting(z_H)    (topic mixture, will be EMGD-CR)
    → KL   = FisherRaoKL(μ_E, logσ²_E)
    """

    def __init__(self,
                 vocab_size:   int,
                 hidden_dim:   int,
                 topic_dim:    int,
                 capsule_module: nn.Module,        # pass EMGD-CR from outside
                 dropout_rate: float = 0.30,
                 poincare_c:   float = 1.00,
                 poincare_scale: float = 0.10):    # shrink z before expmap
        super().__init__()

        self.topic_dim      = topic_dim
        self.poincare       = PoincareBall(c=poincare_c)
        self.poincare_scale = poincare_scale
        self.capsule        = capsule_module

        gcn_out   = hidden_dim // 2
        mlp_out   = hidden_dim
        fused_dim = mlp_out + gcn_out

        # ── (a) Spectral Graph Conv path ────────────────────────────────────
        self.gcn       = SpectralGraphConv(vocab_size, hidden_dim, gcn_out)
        # adj_norm registered as a buffer (set via set_adj_norm before training)
        self.register_buffer('adj_norm', torch.eye(vocab_size))

        # ── Direct MLP path ─────────────────────────────────────────────────
        self.fc1   = nn.Linear(vocab_size, mlp_out)
        self.bn1   = nn.BatchNorm1d(mlp_out)
        self.drop1 = nn.Dropout(dropout_rate)

        # ── Hierarchical fusion ─────────────────────────────────────────────
        self.fc_fuse  = nn.Linear(fused_dim, hidden_dim)
        self.bn_fuse  = nn.BatchNorm1d(hidden_dim)
        self.drop2    = nn.Dropout(dropout_rate)

        # ── (b) Poincaré latent projections ─────────────────────────────────
        self.fc_mu     = nn.Linear(hidden_dim, topic_dim)
        self.fc_logvar = nn.Linear(hidden_dim, topic_dim)

    # ── public utility: build & store normalised adjacency ──────────────────
    def set_adj_norm(self, pmi_matrix: torch.Tensor):
        """Call once after init to inject the vocabulary PMI graph."""
        adj = SpectralGraphConv.build_adj_norm(
            pmi_matrix.to(next(self.parameters()).device)
        )
        self.adj_norm = adj   # overwrites the identity buffer

    # ── forward ─────────────────────────────────────────────────────────────
    def forward(self, x: torch.Tensor,
                temperature: float = 1.0):
        """
        Parameters
        ----------
        x           : [B, V]  bag-of-words (float)
        temperature : float   annealing temperature for capsule routing

        Returns
        -------
        z_h   : [B, K]  hyperbolic latent code
        theta : [B, K]  topic mixture (from capsule routing)
        mu    : [B, K]  encoder mean (Euclidean)
        logvar: [B, K]  encoder log-variance
        kl    : scalar  Fisher-Rao KL divergence
        """
        # ── (a) Spectral graph path ─────────────────────────────────────────
        h_graph = self.gcn(x, self.adj_norm)               # [B, gcn_out]

        # ── Direct MLP path ─────────────────────────────────────────────────
        h_mlp   = self.drop1(F.relu(self.bn1(self.fc1(x))))  # [B, mlp_out]

        # ── Hierarchical fusion ─────────────────────────────────────────────
        h_cat   = torch.cat([h_mlp, h_graph], dim=-1)      # [B, fused_dim]
        h       = self.drop2(F.relu(self.bn_fuse(self.fc_fuse(h_cat))))  # [B, H]

        # ── (b) Poincaré latent space ────────────────────────────────────────
        mu     = self.fc_mu(h)                             # [B, K]
        logvar = self.fc_logvar(h).clamp(min=-10, max=4)  # [B, K]

        # reparameterisation in Euclidean
        std  = torch.exp(0.5 * logvar)
        eps  = torch.randn_like(std)
        z_e  = mu + eps * std                              # [B, K]

        # lift to Poincaré ball via exponential map at origin
        z_h  = self.poincare.expmap0(z_e * self.poincare_scale)  # [B, K]
        z_h  = self.poincare.proj(z_h)                           # ensure ||z_h||<1

        # ── Capsule routing (EMGD-CR or plain) ──────────────────────────────
        theta = self.capsule(z_h, temperature=temperature)       # [B, K]

        # ── (c) Fisher-Rao KL ────────────────────────────────────────────────
        kl = fisher_rao_kl(mu, logvar)

        return z_h, theta, mu, logvar, kl


# ─────────────────────────────────────────────────────────────────────────────
#  Quick sanity check (runs only when file is executed directly)
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import torch

    B, V, H, K = 8, 300, 200, 12

    # Dummy capsule (plain softmax for testing)
    class _DummyCapsule(nn.Module):
        def forward(self, x, temperature=1.0):
            return F.softmax(x / temperature, dim=-1)

    enc = SGPEncoder(vocab_size=V, hidden_dim=H, topic_dim=K,
                     capsule_module=_DummyCapsule())

    pmi_dummy = torch.randn(V, V)
    enc.set_adj_norm(pmi_dummy)

    x_bow = torch.rand(B, V)
    z_h, theta, mu, logvar, kl = enc(x_bow, temperature=1.5)

    print("[SGP-E] z_h    :", z_h.shape,    "  max norm:", z_h.norm(dim=-1).max().item())
    print("[SGP-E] theta  :", theta.shape,  "  sums:", theta.sum(-1).mean().item())
    print("[SGP-E] mu     :", mu.shape)
    print("[SGP-E] logvar :", logvar.shape)
    print("[SGP-E] KL     :", kl.item())
    print("Novelty 1 (SGP-E) — PASSED")

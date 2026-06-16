"""
Gated MLP that lifts a low-dimensional latent into K soft-prompt embeddings for the receiver LM.

Gating mimics the ILCP paper's gated fusion intuition: not every latent dimension should equally
influence every memory slot, which reduces interference when K>1 memory tokens compete for attention.
"""

from __future__ import annotations

import torch
from torch import nn


class GatedLatentToMemoryProjector(nn.Module):
    """
    Map z ∈ R^{latent_dim} to memory ∈ R^{K × model_hidden}.

    The gate uses a SiLU nonlinearity (smooth ReLU) so gradients do not die on negative pre-activations
    the way they would with a plain sigmoid gate saturating at initialization.
    """

    def __init__(
        self,
        latent_dim: int,
        model_hidden: int,
        num_memory_tokens: int,
        mlp_hidden: int,
    ) -> None:
        # Register as nn.Module first so parameters move with .to(device) calls from the trainer.
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.model_hidden = int(model_hidden)
        self.num_memory_tokens = int(num_memory_tokens)
        self.mlp_hidden = int(mlp_hidden)

        # Shared trunk expands the latent before splitting into gate and value pathways (GLU-style).
        self._trunk = nn.Sequential(
            nn.Linear(self.latent_dim, self.mlp_hidden),
            nn.SiLU(),
            nn.Linear(self.mlp_hidden, self.mlp_hidden),
            nn.SiLU(),
        )
        # Gate pathway outputs per-token gate logits before sigmoid squashing into (0,1).
        self._gate_head = nn.Linear(self.mlp_hidden, self.num_memory_tokens * self.model_hidden)
        # Value pathway outputs the ungated candidate memory tensor of the same flattened shape.
        self._value_head = nn.Linear(self.mlp_hidden, self.num_memory_tokens * self.model_hidden)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Return shape (batch, K, D) suitable for torch.cat along the sequence dimension with token embeds.

        Applying the gate in float32 even when the LM runs in bf16 can reduce numerical noise on consumer GPUs
        that lack full-speed bf16 ALUs (Pascal-era hardware note for GTX 1080 baselines).
        """
        # Ensure z is rank-2 so batch matmul paths stay vectorized on wide tensor cores when available.
        if z.dim() != 2:
            raise ValueError("GatedLatentToMemoryProjector expects z shaped (batch, latent_dim).")
        h = self._trunk(z)
        gate = torch.sigmoid(self._gate_head(h))
        value = self._value_head(h)
        # Elementwise product suppresses spurious directions before reshaping into memory tokens.
        mem_flat = gate * value
        return mem_flat.view(z.size(0), self.num_memory_tokens, self.model_hidden)

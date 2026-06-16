"""
β-VAE style compressor for fixed-dimension agent summary embeddings.

The telecom ILCP paper uses a variational bottleneck so transported payloads stay compact while
still regularizing the latent geometry; we mirror that *structure* here without copying RAN metrics.
"""

from __future__ import annotations

import math

import torch
from torch import nn


class BetaVAE(nn.Module):
    """
    Fully-connected β-VAE operating on pooled LM hidden states.

    Why fully-connected instead of convolutions: the input is already a single vector per sample
    (masked mean pool over time), so conv layers would add parameter overhead without exploiting
    spatial locality the way a CUDA kernel would on grids.
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        hidden_dim: int,
        beta: float = 1.0,
    ) -> None:
        # Call Module.__init__ first so submodules register correctly with the autograd graph.
        super().__init__()
        # Store beta outside buffers: it is a hyperparameter toggled by the trainer, not learned.
        self.beta = float(beta)
        # Record topology for checkpoint portability and README-level reproducibility tables.
        self.input_dim = int(input_dim)
        self.latent_dim = int(latent_dim)
        self.hidden_dim = int(hidden_dim)

        # Encoder MLP widens then narrows so the network can form non-linear sufficient statistics
        # before the Gaussian head splits into μ and logσ² parameters.
        self._encoder_body = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.SiLU(),
        )
        # Separate heads keep μ and logvar independently scaled; tying them would constrain curvature.
        self._enc_mu = nn.Linear(self.hidden_dim, self.latent_dim)
        self._enc_logvar = nn.Linear(self.hidden_dim, self.latent_dim)

        # Decoder mirrors encoder width so reconstruction capacity is symmetric around the bottleneck.
        self._decoder_body = nn.Sequential(
            nn.Linear(self.latent_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.input_dim),
        )

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Map a batch of summary embeddings to diagonal-Gaussian parameters.

        Returning logvar instead of std avoids a sqrt during training and improves numerical stability
        when variances become tiny (avoids division blow-ups in KL closed form).
        """
        # Flatten optional middle dimensions so the encoder always sees (batch, input_dim).
        h = self._encoder_body(x)
        mu = self._enc_mu(h)
        logvar = self._enc_logvar(h)
        return mu, logvar

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        z = μ + σ * ε with ε ~ N(0, I); keeps sampling differentiable w.r.t. μ, logvar.

        Using std = exp(0.5 * logvar) pushes log-domain parameters through a smoother mapping than raw std.
        """
        # Draw standard normal noise on the same device/dtype as μ to avoid implicit CPU sync stalls.
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct the high-dimensional summary from a latent code z.

        Decoder outputs live in the same space as pooled LM states so we can use MSE as a sanity loss
        before the projector aligns latents into soft-prompt tokens for the receiver LM.
        """
        return self._decoder_body(z)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Full forward: return reconstruction, μ, logvar, and sampled z for downstream projector loss.

        Returning all four tensors avoids recomputing the encoder when the trainer stitches auxiliary heads.
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decode(z)
        return x_hat, mu, logvar, z

    @staticmethod
    def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Closed-form KL(q(z|x) || N(0,I)) for diagonal Gaussians; averaged over batch for stable logging.

        Summing across latent dims matches the standard VAE objective while .mean() keeps scale stable
        when latent_dim changes between ablations without retuning the outer learning rate.
        """
        # -0.5 * sum(1 + logσ² - μ² - exp(logσ²)) is the analytic expectation under the encoder.
        latent_kl = -0.5 * torch.sum(1.0 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        return latent_kl.mean()

    def loss_components(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Convenience bundle: reconstruction + β * KL so train.py stays orchestration-only.

        Using F.mse_loss reduction mean keeps gradient magnitude comparable across batch sizes on wide GPUs.
        """
        x_hat, mu, logvar, _z = self.forward(x)
        recon = torch.nn.functional.mse_loss(x_hat, x, reduction="mean")
        kl = self.kl_divergence(mu, logvar)
        total = recon + self.beta * kl
        return {"total": total, "recon": recon, "kl": kl}


def latent_payload_bytes(latent_dim: int, dtype: torch.dtype) -> int:
    """
    Report transferable payload size in bytes for README receipts (not assumed 128-byte telecom payload).

    Element size follows torch.dtype element alignment; this is the on-wire analog for in-process transport.
    """
    # torch.finfo / element_size gives byte width for floating dtypes used in z tensors.
    if not dtype.is_floating_point:
        raise ValueError("latent_payload_bytes expects a floating dtype for z.")
    width = torch.tensor([], dtype=dtype).element_size()
    return int(latent_dim) * int(width)

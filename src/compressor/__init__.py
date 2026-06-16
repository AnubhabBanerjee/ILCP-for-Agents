# Package marker for the learned state compressor (β-VAE / information bottleneck).
# Keeping __init__ exports narrow avoids accidental deep import cycles during agent harness startup.
from compressor.beta_vae import BetaVAE

__all__ = ["BetaVAE"]

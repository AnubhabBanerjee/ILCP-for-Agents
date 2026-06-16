import torch

from compressor.beta_vae import BetaVAE, latent_payload_bytes
from projector.gated_mlp import GatedLatentToMemoryProjector
from transport.in_process import InProcessTransport


def test_beta_vae_shapes_and_loss():
    """
    Ensure the VAE forward path preserves batch geometry and returns finite loss terms without a GPU.

    CPU-only execution keeps GitHub Actions cheap while still validating tensor rank contracts the harness relies on.
    """
    torch.manual_seed(0)
    b, d, lat = 4, 32, 8
    vae = BetaVAE(input_dim=d, latent_dim=lat, hidden_dim=64, beta=1e-3)
    x = torch.randn(b, d)
    out = vae.loss_components(x)
    assert torch.isfinite(out["total"]).item()
    x_hat, mu, logvar, z = vae.forward(x)
    assert x_hat.shape == x.shape
    assert z.shape == (b, lat)


def test_projector_gating_shapes():
    """
    Validate memory bank reshaping because downstream concat assumes (batch, K, D) not flattened layouts.

    A mistaken view() would still run but silently scramble token order when prepended to Qwen embeddings.
    """
    torch.manual_seed(0)
    z = torch.randn(2, 16)
    proj = GatedLatentToMemoryProjector(latent_dim=16, model_hidden=32, num_memory_tokens=3, mlp_hidden=64)
    mem = proj(z)
    assert mem.shape == (2, 3, 32)


def test_transport_round_trip_device_move():
    """
    Confirm pack/unpack preserves numeric bytes across a CPU staging boundary mimicking an Xn copy.

    Using float32 here isolates dtype bugs from bfloat16 rounding noise during tight equality asserts.
    """
    transport = InProcessTransport(device="cpu")
    z = torch.randn(5, dtype=torch.float32)
    payload = transport.pack(z)
    z2 = transport.unpack(payload, device=torch.device("cpu"))
    assert torch.allclose(z, z2.cpu())
    assert latent_payload_bytes(5, torch.float32) == 20


def test_masked_mean_pool_logic():
    """
    Directly test pooling math independent of transformers to guard the approved V1 summary embedding.

    Padding rows must contribute zero mass so shorter sequences are not diluted by padded zeros incorrectly.
    """
    from agents.qwen_encoder import QwenContextEncoder

    hidden = torch.tensor(
        [
            [[1.0, 0.0], [3.0, 0.0], [0.0, 0.0]],
            [[2.0, 2.0], [0.0, 0.0], [0.0, 0.0]],
        ]
    )
    mask = torch.tensor([[1, 1, 0], [1, 0, 0]], dtype=torch.long)
    pooled = QwenContextEncoder.masked_mean_pool(hidden, mask)
    assert torch.allclose(pooled[0], torch.tensor([2.0, 0.0]))
    assert torch.allclose(pooled[1], torch.tensor([2.0, 2.0]))

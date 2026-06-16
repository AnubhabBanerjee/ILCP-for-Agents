#!/usr/bin/env python3
"""
Train the β-VAE compressor and gated projector on pooled Qwen summary embeddings.

Encoder weights stay frozen so VRAM is dominated by one forward at a time, mirroring deployment where
Agent A already paid the prefill cost and only ships a compact latent across the hop boundary.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch import nn
from tqdm import tqdm

# Allow `python scripts/train.py` without manual PYTHONPATH exports during local iteration loops.
_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from agents.qwen_encoder import QwenContextEncoder
from compressor.beta_vae import BetaVAE
from projector.gated_mlp import GatedLatentToMemoryProjector


def _load_texts(path: Path) -> list[str]:
    """
    Load JSON list of hand-off examples and concatenate context+question as pseudo Agent A windows.

    Concatenation inflates token length compared to context-only pooling but keeps the training signal
    anchored to the same passages the benchmark harness will later score without maintaining two files.
    """
    raw = json.loads(path.read_text(encoding="utf-8"))
    texts: list[str] = []
    for row in raw:
        ctx = str(row["context"])
        q = str(row["question"])
        texts.append(f"{ctx}\n\nQuestion: {q}")
    return texts


def main() -> None:
    """
    CLI entry: parse knobs, materialize modules, optimize with AdamW, persist a fused checkpoint dict.

    argparse defaults intentionally mirror benchmark_campaign.py.example so sweeps do not desync shapes.
    """
    parser = argparse.ArgumentParser(description="Train ILCP compressor + projector for ilcp-for-agents.")
    parser.add_argument("--data", type=Path, default=_ROOT / "data" / "toy_handoff.json")
    parser.add_argument("--out", type=Path, default=_ROOT / "checkpoints" / "ilcp_stage1.pt")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--vae-hidden", type=int, default=2048)
    parser.add_argument("--proj-hidden", type=int, default=2048)
    parser.add_argument("--memory-tokens", type=int, default=4)
    parser.add_argument("--beta", type=float, default=1.0e-3)
    parser.add_argument("--align-weight", type=float, default=5.0)
    parser.add_argument("--lr", type=float, default=1.0e-4)
    parser.add_argument("--max-length", type=int, default=512)
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    texts = _load_texts(args.data)
    encoder = QwenContextEncoder(device_map="auto" if torch.cuda.is_available() else None)
    encoder.eval()
    for p in encoder.lm.parameters():
        p.requires_grad_(False)

    d_model = encoder.hidden_size
    vae = BetaVAE(input_dim=d_model, latent_dim=args.latent_dim, hidden_dim=args.vae_hidden, beta=args.beta).to(
        device
    )
    proj = GatedLatentToMemoryProjector(
        latent_dim=args.latent_dim,
        model_hidden=d_model,
        num_memory_tokens=args.memory_tokens,
        mlp_hidden=args.proj_hidden,
    ).to(device)

    opt = torch.optim.AdamW(list(vae.parameters()) + list(proj.parameters()), lr=args.lr)
    mse = nn.functional.mse_loss

    args.out.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        total_loss = 0.0
        pbar = tqdm(texts, desc=f"epoch {epoch+1}/{args.epochs}")
        for text in pbar:
            pooled, _mask = encoder.encode_contexts([text], max_length=args.max_length)
            x = pooled.detach().to(device)
            x_hat, mu, logvar, z = vae(x)
            recon = mse(x_hat, x)
            kl = vae.kl_divergence(mu, logvar)
            mem = proj(z)
            mean_mem = mem.mean(dim=1)
            align = mse(mean_mem, x)
            loss = recon + args.beta * kl + args.align_weight * align
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total_loss += float(loss.detach().cpu())
            pbar.set_postfix(loss=float(loss.detach().cpu()))
        print(f"epoch {epoch+1} mean loss={total_loss / max(len(texts),1):.6f}")

    torch.save(
        {
            "vae": vae.state_dict(),
            "projector": proj.state_dict(),
            "meta": {
                "latent_dim": args.latent_dim,
                "vae_hidden": args.vae_hidden,
                "proj_hidden": args.proj_hidden,
                "memory_tokens": args.memory_tokens,
                "beta": args.beta,
                "align_weight": args.align_weight,
            },
        },
        args.out,
    )
    print(f"wrote checkpoint to {args.out}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Smoke-run the ILCP harness on the toy JSON dataset (requires GPU RAM for 7B unless ILCP_MODEL_ID overrides).

The demo prints timing and exact-match scores so developers can sanity-check wiring before launching a
full three-trial benchmark campaign required for blog receipts.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from agents.harness import IlcpAgentHarness, ToyHandoffExample, load_ilcp_modules
from agents.qwen_encoder import QwenContextEncoder
from transport.in_process import InProcessTransport


def _load_examples(path: Path) -> list[ToyHandoffExample]:
    """
    Mirror benchmark JSON parsing so demo and campaign scripts cannot drift silently on schema tweaks.

    Returning a list keeps the demo deterministic regardless of pandas availability on edge laptops.
    """
    rows = json.loads(path.read_text(encoding="utf-8"))
    return [ToyHandoffExample(context=str(r["context"]), question=str(r["question"]), answer=str(r["answer"])) for r in rows]


def main() -> None:
    """
    Load optional checkpoint, construct harness, and print one cold vs ILCP comparison for the first example.

    Stopping after the first row avoids multi-minute wall times when someone runs the demo interactively
    on a thermally constrained mobile GPU without realizing the full suite would queue many forwards.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = _ROOT / "checkpoints" / "ilcp_stage1.pt"
    ckpt = None
    if ckpt_path.exists():
        try:
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        except TypeError:
            ckpt = torch.load(ckpt_path, map_location="cpu")

    encoder = QwenContextEncoder(device_map="auto" if torch.cuda.is_available() else None)
    meta = (ckpt or {}).get("meta", {})
    vae, proj = load_ilcp_modules(
        encoder=encoder,
        latent_dim=int(meta.get("latent_dim", 128)),
        vae_hidden=int(meta.get("vae_hidden", 2048)),
        projector_mlp_hidden=int(meta.get("proj_hidden", 2048)),
        num_memory_tokens=int(meta.get("memory_tokens", 4)),
        checkpoint=ckpt,
        device=device,
    )
    transport = InProcessTransport(device="cpu")
    harness = IlcpAgentHarness(encoder=encoder, vae=vae, projector=proj, transport=transport)

    examples = _load_examples(_ROOT / "data" / "toy_handoff.json")
    ex = examples[0]
    cold_text, cold_dt = harness.run_cold_baseline(ex)
    ilcp_text, ilcp_dt = harness.run_ilcp_branch(ex)
    print("=== ilcp-for-agents demo (first toy example) ===")
    print(f"cold latency_s={cold_dt:.4f} exact={harness.quality_exact_match(ex, cold_text)} text={cold_text[:200]!r}")
    print(f"ilcp latency_s={ilcp_dt:.4f} exact={harness.quality_exact_match(ex, ilcp_text)} text={ilcp_text[:200]!r}")
    if ckpt is None:
        print("note: no checkpoint found; compressor/projector are randomly initialized for wiring smoke only.")


if __name__ == "__main__":
    main()

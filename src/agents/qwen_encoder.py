"""
Qwen2.5 encoder utilities with approved V1 pooling: masked mean of final-layer hidden states.

We intentionally avoid returning every layer's activations: PCIe-style host churn and VRAM pressure
both scale linearly with depth, but the ILCP analogy only needs one fixed-size summary vector per hop.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterable

import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from transformers import BitsAndBytesConfig as _BitsAndBytesConfig
except Exception:  # pragma: no cover - transformers always ships the symbol; guard keeps mypy/runtime symmetry.
    _BitsAndBytesConfig = None  # type: ignore[misc, assignment]


@dataclass(frozen=True)
class QwenEncoderConfig:
    """
    Immutable load-time knobs so train.py and harness.py share identical LM wiring without globals.

    Using an env override keeps CI laptops able to point at tiny models while GTX 1080 rigs stay on 7B.
    """

    model_id: str = "Qwen/Qwen2.5-7B-Instruct"
    trust_remote_code: bool = True


def default_model_id() -> str:
    """
    Resolve which HF repo id to load, honoring ILCP_MODEL_ID without silently changing series defaults.

    The blog series names the GGUF Q4_K_M artifact for llama.cpp; HF weights are the practical path for
    autograd-friendly hidden states, so we document both in README instead of pretending byte parity.
    """
    return os.environ.get("ILCP_MODEL_ID", "Qwen/Qwen2.5-7B-Instruct")


def _try_build_4bit_config():
    """
    Construct NF4 load flags when bitsandbytes kernels are available on the active CUDA device.

    Pascal (GTX 1080, sm_61) may fail at runtime: callers fall back to fp16 or CPU per README guidance.
    """
    if _BitsAndBytesConfig is None:
        return None
    try:
        return _BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
    except Exception:
        return None


class QwenContextEncoder(nn.Module):
    """
    Thin wrapper around AutoModelForCausalLM exposing pooled context embeddings for ILCP.

    nn.Module subclassing lets optimizers optionally treat this container as part of a larger module tree
    even though we mostly call forward for inference-style encoding during data collection.
    """

    def __init__(self, model_id: str | None = None, device_map: str | dict | None = "auto") -> None:
        super().__init__()
        mid = model_id or default_model_id()
        self.model_id = mid
        self.tokenizer = AutoTokenizer.from_pretrained(mid, trust_remote_code=True)
        # Qwen instruct checkpoints often ship without an explicit pad token; eos-as-pad keeps batching valid.
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        bnb_cfg = None
        if torch.cuda.is_available():
            bnb_cfg = _try_build_4bit_config()
        try:
            if bnb_cfg is not None:
                self.lm = AutoModelForCausalLM.from_pretrained(
                    mid,
                    quantization_config=bnb_cfg,
                    device_map=device_map,
                    trust_remote_code=True,
                )
            else:
                self.lm = AutoModelForCausalLM.from_pretrained(
                    mid,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    device_map=device_map if torch.cuda.is_available() else None,
                    trust_remote_code=True,
                )
        except Exception as exc:
            raise RuntimeError(
                "Failed to load Qwen weights. On 8GB Pascal GPUs, 4-bit bitsandbytes may be unavailable; "
                "set ILCP_MODEL_ID to a smaller instruct model or use a machine with more VRAM. "
                f"Original error: {exc}"
            ) from exc
        self.hidden_size = int(self.lm.config.hidden_size)

    @property
    def device(self) -> torch.device:
        """
        Infer the primary parameter device for tensor staging without assuming a single .cuda() index.

        device_map='auto' spreads layers; the first parameter device is a reasonable anchor for z staging.
        """
        return next(self.lm.parameters()).device

    @staticmethod
    def masked_mean_pool(last_hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Approved V1 pooling: average final-layer token vectors with padding masked to zero mass.

        Dividing by the raw token count (not L2-normalizing) preserves magnitude cues about confidence
        and saturation that a unit-norm pool would erase before the VAE bottleneck.
        """
        if last_hidden.dim() != 3:
            raise ValueError("last_hidden must be (batch, seq, dim).")
        if attention_mask.dim() != 2:
            raise ValueError("attention_mask must be (batch, seq).")
        mask = attention_mask.unsqueeze(-1).to(dtype=last_hidden.dtype, device=last_hidden.device)
        summed = (last_hidden * mask).sum(dim=1)
        lengths = mask.sum(dim=1).clamp(min=1.0)
        return summed / lengths

    @torch.inference_mode()
    def encode_contexts(
        self,
        texts: list[str],
        max_length: int = 512,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Return pooled embeddings (batch, hidden) and the attention masks used for auditing shapes.

        torch.inference_mode() disables version counter bookkeeping entirely vs no_grad for slightly lower overhead
        when sweeping thousands of contexts for compressor dataset construction on a budget GPU.
        """
        batch = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        batch = {k: v.to(self.device) for k, v in batch.items()}
        outputs = self.lm(**batch, output_hidden_states=True, use_cache=False)
        last_hidden = outputs.hidden_states[-1]
        pooled = self.masked_mean_pool(last_hidden, batch["attention_mask"])
        return pooled, batch["attention_mask"]

    @torch.inference_mode()
    def pooled_embedding_for_text(self, text: str, max_length: int = 512) -> torch.Tensor:
        """
        Convenience single-string helper so demo.py stays readable without manual batch unsqueeze calls.

        Returning only the vector (not the mask) is sufficient when downstream code logs seq lengths separately.
        """
        vec, _mask = self.encode_contexts([text], max_length=max_length)
        return vec[0]

    def get_input_embeddings(self) -> nn.Module:
        """
        Expose the token embedding matrix for constructing inputs_embeds prefixes in the ILCP branch.

        Directly reusing the LM's embedding table keeps memory tokens in the same vector space as real tokens
        so attention kernels see a unified semantic geometry instead of an ad-hoc parallel embedding space.
        """
        return self.lm.get_input_embeddings()

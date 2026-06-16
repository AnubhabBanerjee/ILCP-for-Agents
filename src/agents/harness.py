"""
Multi-hop harness comparing a cold-start receiver against the ILCP compress→transport→project path.

Timing uses perf_counter (host wall time) instead of CUDA events here so benchmarks remain comparable
when part of the model is CPU-offloaded or when drivers lack async event support on older stacks.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import torch
from torch import nn

from agents.qwen_encoder import QwenContextEncoder
from compressor.beta_vae import BetaVAE
from projector.gated_mlp import GatedLatentToMemoryProjector
from transport.in_process import InProcessTransport


@dataclass(frozen=True)
class ToyHandoffExample:
    """
    Minimal record for one A→B hop: long context, short question, reference answer string.

    Freezing keeps hashes stable when examples are logged into CSV receipts across benchmark trials.
    """

    context: str
    question: str
    answer: str


def _format_agent_prompt(context: str, question: str) -> str:
    """
    Build the cold-start string that forces the receiver LM to re-read the entire Agent A context.

    Keeping the instruction delimiter style stable across branches isolates the ILCP effect from prompt drift.
    """
    return (
        "You are Agent B. Read the context carefully, then answer the question with a short span.\n\n"
        f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
    )


def _format_question_only_prompt(question: str) -> str:
    """
    ILCP branch prompt: deliberately omits the raw context so any competence must flow through memory tokens.

    If this string accidentally included the passage, receipts would lie about skipping re-prefill work.
    """
    return (
        "You are Agent B. You do not see the original passage; rely on the prepended memory embeddings.\n\n"
        f"Question: {question}\n\nAnswer:"
    )


def _normalize_answer(text: str) -> str:
    """
    Collapse whitespace and case for a strict toy exact-match metric (not a general NLP benchmark).

    Normalization avoids penalizing trivial tokenizer whitespace deltas while staying deterministic.
    """
    return " ".join(text.strip().lower().split())


@torch.inference_mode()
def greedy_generate_from_ids(
    lm: nn.Module,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    device: torch.device,
) -> str:
    """
    Standard greedy decoding for the cold-start baseline using token ids end-to-end.

    torch.inference_mode minimizes autograd overhead on the baseline path so latency comparisons
    against the ILCP prefix path are not systematically biased by grad bookkeeping.
    """
    batch = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    batch = {k: v.to(device) for k, v in batch.items()}
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    out_ids = lm.generate(
        **batch,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=pad_id,
        use_cache=True,
    )
    gen = out_ids[0, batch["input_ids"].shape[1] :]
    return tokenizer.decode(gen, skip_special_tokens=True)


@torch.inference_mode()
def greedy_generate_from_memory_prefix(
    lm: nn.Module,
    tokenizer,
    embed_layer: nn.Module,
    memory_tokens: torch.Tensor,
    question_prompt: str,
    max_new_tokens: int,
    device: torch.device,
) -> str:
    """
    Greedy decoding starting from concatenated soft prompts + question token embeddings.

    The loop alternates between a wide prefix forward (first step) and skinny single-token steps that
    reuse KV cache entries the way a production decoder would after a hand-off frame arrives.
    """
    q_batch = tokenizer(question_prompt, return_tensors="pt", truncation=True, max_length=512)
    q_ids = q_batch["input_ids"].to(device)
    q_embeds = embed_layer(q_ids)
    prefix = torch.cat([memory_tokens.unsqueeze(0), q_embeds], dim=1)
    attention_mask = torch.ones(prefix.shape[:2], device=device, dtype=torch.long)
    past = None
    embed_step = prefix
    generated: list[int] = []
    for _ in range(max_new_tokens):
        out = lm(
            inputs_embeds=embed_step,
            attention_mask=attention_mask,
            past_key_values=past,
            use_cache=True,
            return_dict=True,
        )
        past = out.past_key_values
        logits = out.logits[:, -1, :]
        next_id = int(logits.argmax(dim=-1).item())
        generated.append(next_id)
        next_emb = embed_layer(torch.tensor([[next_id]], device=device, dtype=torch.long))
        embed_step = next_emb
        add = torch.ones((1, 1), device=device, dtype=torch.long)
        attention_mask = torch.cat([attention_mask, add], dim=1)
    return tokenizer.decode(generated, skip_special_tokens=True)


class IlcpAgentHarness:
    """
    Wires Qwen + VAE + projector + transport for side-by-side cold vs ILCP evaluation runs.

    The harness keeps modules in eval() during inference so dropout cannot inject non-reproducible noise
    into latency p99 measurements across the three mandated trials.
    """

    def __init__(
        self,
        encoder: QwenContextEncoder,
        vae: BetaVAE,
        projector: GatedLatentToMemoryProjector,
        transport: InProcessTransport,
    ) -> None:
        # Stash references on self without cloning weights so checkpoints reload in O(1) file descriptor time.
        self.encoder = encoder
        self.vae = vae
        self.projector = projector
        # Persist the transport object so benchmarks can swap in networked implementations later without API churn.
        self.transport = transport
        self.lm = encoder.lm
        self.tokenizer = encoder.tokenizer
        self.embed_layer = encoder.get_input_embeddings()

    @torch.inference_mode()
    def encode_sender_summary(self, context: str) -> torch.Tensor:
        """
        Agent A path: pooled hidden-state summary embedding s_A used as the VAE input tensor.

        Returning a detached vector is not required here because inference_mode already blocks autograd,
        but callers may log norms without risking accidental graph retention across hops.
        """
        return self.encoder.pooled_embedding_for_text(context)

    @torch.inference_mode()
    def ilcp_memory_from_context(self, context: str, device: torch.device) -> torch.Tensor:
        """
        Compress→transport→project pipeline returning receiver memory tensor (K, D) on `device`.

        Using the VAE encoder mean μ (not a stochastic sample) stabilizes multi-trial latency and quality
        comparisons the way a deployed system would freeze stochasticity after calibration.
        """
        s_a = self.encode_sender_summary(context).to(device)
        mu, _logvar = self.vae.encode(s_a.unsqueeze(0))
        z = mu.squeeze(0)
        payload = self.transport.pack(z)
        z_b = self.transport.unpack(payload, device=device)
        mem = self.projector(z_b.unsqueeze(0)).squeeze(0)
        return mem

    def run_cold_baseline(self, ex: ToyHandoffExample, max_new_tokens: int = 48) -> tuple[str, float]:
        """
        Receiver rebuilds from raw tokens; returns decoded tail and wall seconds for receipts CSV rows.

        perf_counter includes Python overhead on purpose: blog readers care about end-to-end hop latency,
        not an unrealistic kernel-only stopwatch that omits tensor staging and tokenizer work.
        """
        device = self.encoder.device
        prompt = _format_agent_prompt(ex.context, ex.question)
        t0 = time.perf_counter()
        text = greedy_generate_from_ids(self.lm, self.tokenizer, prompt, max_new_tokens, device)
        dt = time.perf_counter() - t0
        return text, dt

    def run_ilcp_branch(self, ex: ToyHandoffExample, max_new_tokens: int = 48) -> tuple[str, float]:
        """
        Receiver starts from transported latent only; question text is visible but the passage is not.

        The timer starts before compression so the receipt includes sender-side pooling + VAE encoder work
        that would execute on Agent A silicon in a distributed deployment story.
        """
        device = self.encoder.device
        t0 = time.perf_counter()
        mem = self.ilcp_memory_from_context(ex.context, device=device)
        q_prompt = _format_question_only_prompt(ex.question)
        text = greedy_generate_from_memory_prefix(
            self.lm,
            self.tokenizer,
            self.embed_layer,
            mem,
            q_prompt,
            max_new_tokens,
            device,
        )
        dt = time.perf_counter() - t0
        return text, dt

    def quality_exact_match(self, ex: ToyHandoffExample, model_text: str) -> float:
        """
        Return 1.0 when normalized model output equals normalized reference answer, else 0.0.

        Exact match is a harsh metric on generative answers but is trivial to audit in CSVs without BLEU deps.
        """
        return 1.0 if _normalize_answer(model_text) == _normalize_answer(ex.answer) else 0.0


def load_ilcp_modules(
    encoder: QwenContextEncoder,
    latent_dim: int,
    vae_hidden: int,
    projector_mlp_hidden: int,
    num_memory_tokens: int,
    checkpoint: dict | None,
    device: torch.device,
) -> tuple[BetaVAE, GatedLatentToMemoryProjector]:
    """
    Factory that rebuilds train.py-shaped modules and optionally loads synchronized state_dict blobs.

    Keeping construction centralized avoids shape drift between training and benchmarking when latent_dim
    changes during research without updating every script argparse default in lockstep.
    """
    d_model = encoder.hidden_size
    vae = BetaVAE(input_dim=d_model, latent_dim=latent_dim, hidden_dim=vae_hidden).to(device)
    proj = GatedLatentToMemoryProjector(
        latent_dim=latent_dim,
        model_hidden=d_model,
        num_memory_tokens=num_memory_tokens,
        mlp_hidden=projector_mlp_hidden,
    ).to(device)
    if checkpoint is not None:
        vae.load_state_dict(checkpoint["vae"])
        proj.load_state_dict(checkpoint["projector"])
    vae.eval()
    proj.eval()
    return vae, proj

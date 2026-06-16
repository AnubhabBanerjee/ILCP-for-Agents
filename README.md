**Copyright (c) 2026 Anubhab Banerjee (AnubhabBanerjee/ilcp-for-agents)** **All rights reserved. No part of this repository may be used, redistributed, or modified in any form or by any means without the prior written permission of the author.**

---
 

---

# 🚀 ILCP for Agents: latent hand-offs for multi-hop LLM pipelines

> **PyTorch + HuggingFace Qwen harness:** learn a **compress → transport → project** path so a receiving agent can prepend **soft-prompt memory** instead of cold-starting through a full context rebuild. The telecom ILCP paper is a **conceptual anchor only**—**no RAN / drive-test numbers from that paper are reused as LLM-agent results here.**

**This repository is Part 4** of the *Production-Grade Agentic Inference* series (*Towards Data Science*). Please see below for other parts in this series.

## 🔗 Series links

- **Part 1 — SwarmKV**

Full deep-dive: `https://towardsdatascience.com/kv-cache-reuse-for-multi-agent-llm-inference-i-built-a-c-orchestrator-so-my-gpu-would-stop-reading-the-same-document-twice/`; 

Repository link: `https://github.com/AnubhabBanerjee/swarmkv`

- **Part 2 — Kube-Timeslice-Profiler**

Full deep-dive: `https://towardsdatascience.com/gpu-time-slicing-for-concurrent-llm-agents-on-kubernetes/`

Repository link: `https://github.com/AnubhabBanerjee/Kube-Timeslice-Profiler`

- **Part 3 — CUDA-TopK-Retrieval**

Full deep-dive: `TBA`

Repository link: `https://github.com/AnubhabBanerjee/CUDA-TopK-Retrieval`

- **Part 4 — this repository** 

Full deep-dive: `TBA`

Repository link: `TBA`

## 🧠 System architecture

End-to-end flow (V1, in-process transport):

1. **Agent A (sender)** — tokenizes its working context through **Qwen2.5-7B-Instruct** (HF checkpoint aligned with the series **GGUF** `Qwen2.5-7B-Instruct-Q4_K_M`; see stack notes below), takes the **final hidden layer**, and forms a **fixed-size summary** \(s_A\) via **masked mean pooling** over all non-padding tokens (approved V1 definition).
2. **Compressor (β-VAE spirit)** — maps \(s_A\) to a low-dimensional latent **z** with reconstruction + KL pressure so the payload stays an information bottleneck.
3. **Transport (V1)** — serializes **z** to an explicit **`TransportPayload`** (CPU-staged bytes) as the analog of a telecom hand-over record; no network socket yet.
4. **Projector (gated MLP)** — lifts **z** into **K** memory vectors in the LM embedding space so the receiver can **concat** them ahead of question token embeddings.
5. **Agent B (receiver)** — **cold baseline:** full context + question in `input_ids`; **ILCP branch:** question-only text plus the projected prefix in **`inputs_embeds`**, then greedy continuation for benchmarking.

## 🛠️ Stack & reference targets

| Layer | Role |
|-------|------|
| **PyTorch + transformers** | Training/eval for compressor, projector, and LM-backed harness (Part 4 override per `Instructions_for_cursor_agent.md`). |
| **Qwen / Qwen2.5-7B-Instruct (HF)** | Hidden states + generation; override with **`ILCP_MODEL_ID`** when VRAM is insufficient. |
| **Canonical series GPU** | **NVIDIA GTX 1080** (Pascal **sm_61**, **8 GB**) for headline receipts when available. |

**GGUF vs HF:** the blog series cites **`Qwen2.5-7B-Instruct-Q4_K_M.gguf`** for llama.cpp-style workflows. This repo uses **HuggingFace weights** for **autograd-friendly** pooled states; **do not assume byte-for-byte parity** with the GGUF artifact—document what you actually ran in `examples/example-run-results/results_narrative.txt`.

## 🧬 Conceptual anchor (telecom paper — cite, do not copy numbers)

- **Title:** *Inductive Latent Context Persistence: Closing the Post-Handover Cold Start in 6G Radio Access Networks.*
- **Authors:** Anubhab Banerjee & Daniyal Amir Awan (Nokia Solutions and Networks, Munich)
- **Venue:** accepted at **AI4NextG @ ICML 2026**
- **PDF:** https://arxiv.org/pdf/2605.00593

## ✅ Prerequisites

* **Python 3.10+** (3.12+ recommended to match sibling repos’ venv stories).
* **GPU** strongly recommended for **7B** training/inference; **8 GiB Pascal** may require **`ILCP_MODEL_ID`** to a smaller instruct model or CPU/offload paths—disclose whatever you used in receipts.
* **HF access** to download **`Qwen/Qwen2.5-7B-Instruct`** (or your override id) the first time you train or benchmark.

## ⚙️ Installation

```bash
git clone <github-repo-url> ilcp-for-agents
cd ilcp-for-agents
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

## 🚀 Execution

**Train** the compressor + projector (writes `checkpoints/ilcp_stage1.pt` by default):

```bash
python3 scripts/train.py
```

**Benchmark** (copy the template first—matches the SwarmKV / CUDA-TopK convention):

```bash
cp scripts/benchmark_campaign.py.example scripts/benchmark_campaign.py
python3 scripts/benchmark_campaign.py
python3 scripts/plot_results.py
```

**Smoke demo** (first toy row only; requires a loadable model):

```bash
PYTHONPATH=src python3 examples/demo.py
```

Artifacts land under **`examples/example-run-results/`** (see that folder’s `README.md`).

## 🎬 Example run

After you have a checkpoint and successful benchmark:

```text
$ cp scripts/benchmark_campaign.py.example scripts/benchmark_campaign.py
$ python3 scripts/benchmark_campaign.py
wrote .../examples/example-run-results/all_trials.csv and best_run.json

$ python3 scripts/plot_results.py
wrote .../examples/example-run-results/plots/latency_cold.png
wrote .../examples/example-run-results/plots/latency_ilcp.png
```

## 📁 Project layout

```text
ilcp-for-agents/
├── README.md
├── requirements.txt
├── data/
│   └── toy_handoff.json              # tiny wiring examples (replace for real receipts)
├── src/
│   ├── compressor/                   # β-VAE-style bottleneck
│   ├── projector/                    # gated MLP → K memory embeddings
│   ├── transport/                    # in-process payload pack/unpack
│   └── agents/                       # Qwen encoder + cold vs ILCP harness
├── scripts/
│   ├── train.py
│   ├── benchmark_campaign.py.example
│   └── plot_results.py
├── examples/
│   └── demo.py
└── tests/
```


## 🧱 LIMITATIONS (honest caveats)

- **Lossy state:** V1 moves a **pooled hidden summary**, not full activations or KV tensors—genuine risk of dropped detail; say so in any write-up.
- **Toy metric:** default **exact match** against `data/toy_handoff.json` is a **harsh wiring check**, not a claim about open-domain QA quality.
- **In-process transport:** no real cross-host wire protocol yet; payload bytes are still reported for the “compact record” story.
- **Pascal + bitsandbytes:** NF4 loading may be unavailable or unstable on some **sm_61** stacks—fall back or change **`ILCP_MODEL_ID`**, then document.
- **CI vs GPU:** unit tests validate tensor shapes/pooling math **without** downloading **7B** weights; GPU receipts remain a local responsibility.

## 🛣️ Roadmap

* **Real task + held-out metric** aligned with your publication target (replace toy JSON and tighten the quality contract).
* **Optional secondary GPU** runs documented alongside the **GTX 1080** canonical row.
* **KV-derived ILCP** (README-only future work unless scoped)—closes the loop with **SwarmKV** but is a research-heavy direction.


## 🙏 Acknowledgments

Built with **PyTorch**, **HuggingFace transformers**, and the **Qwen** model family. Model weights and upstream licenses remain with their respective licensors; this repository does not redistribute GGUF weights.

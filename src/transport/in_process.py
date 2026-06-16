"""
V1 transport layer: an explicit serialize/deserialize path for z even when both agents share RAM.

Having a dedicated payload type keeps future networking (gRPC, shared memory ring buffers) behind a
stable interface without rewriting the projector or compressor call sites.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class TransportPayload:
    """
    Immutable container for the latent tensor plus minimal metadata for audit logs.

    Freezing the dataclass prevents accidental in-place mutation that would desync byte-size receipts
    between the sending and receiving agent in multi-threaded harnesses.
    """

    latent: torch.Tensor
    dtype_name: str
    latent_dim: int

    def byte_length(self) -> int:
        """
        Compute exact serialized byte length for README tables (torch.save uses pickle; we use raw bytes).

        Raw contiguous bytes mirror the "payload over Xn" story more honestly than pickling full tensors.
        """
        return int(self.latent.numel() * self.latent.element_size())


class InProcessTransport:
    """
    Serialize z to CPU bytes and back; analog of copying a compact ILCP record across an agent boundary.

    Moving to CPU before measuring byte_length avoids counting GPU allocator padding that can vary by driver.
    """

    def __init__(self, device: torch.device | str = "cpu") -> None:
        # Default staging device is CPU so payload sizes match host-visible transfers in microbenchmarks.
        self.device = torch.device(device)

    def pack(self, z: torch.Tensor) -> TransportPayload:
        """
        Detach from autograd, move to staging device, and record dtype for round-trip fidelity.

        Detaching breaks the graph on purpose: transport is a hand-off boundary where sender gradients stop.
        """
        z_staged = z.detach().to(self.device).contiguous()
        return TransportPayload(
            latent=z_staged,
            dtype_name=str(z_staged.dtype),
            latent_dim=int(z_staged.shape[-1]),
        )

    def unpack(self, payload: TransportPayload, device: torch.device | str) -> torch.Tensor:
        """
        Materialize z on the receiver device before the projector lifts it into memory embeddings.

        clone() prevents aliasing if the same payload object were accidentally reused across two agents.
        """
        return payload.latent.clone().to(device, non_blocking=True)

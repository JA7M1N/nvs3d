"""Positional encoding for NeRF.

(verify): γ(p) = [p, sin(2⁰πp), cos(2⁰πp), …, sin(2^{L-1}πp), cos(2^{L-1}πp)]
          — blueprint line 149.
(verify): L=10 for position (dim 63), L=4 for direction (dim 27) — line 149.
(verify): Frequencies use π: freq_l = 2^l · π — line 149.

Layout: For D-dimensional input, output is
  [p₁..pD, sin(f₀·p₁)..sin(f₀·pD), cos(f₀·p₁)..cos(f₀·pD), …]
  Total dim = D·(1 + 2L)  if include_input, else D·2L.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
from torch import Tensor


class PositionalEncoding(nn.Module):
    """Positional encoding: maps input to a higher-dimensional space.

    Frequencies: freq_l = 2^l · π for l = 0, 1, …, L-1.

    Args:
        L: Number of frequency bands.
        include_input: If True, prepend the raw input to the output.
    """

    def __init__(self, L: int, include_input: bool = True) -> None:
        super().__init__()
        self.L = L
        self.include_input = include_input

        # Precompute frequency bands: 2^0·π, 2^1·π, …, 2^{L-1}·π
        # Stored in float64 to preserve precision for float64 inputs.
        freqs = torch.tensor([2.0 ** l * math.pi for l in range(L)], dtype=torch.float64)
        self.register_buffer("freqs", freqs)  # [L]

    def output_dim(self, input_dim: int) -> int:
        """Compute the output dimension for a given input dimension."""
        d = input_dim * 2 * self.L
        if self.include_input:
            d += input_dim
        return d

    def forward(self, x: Tensor) -> Tensor:
        """Apply positional encoding.

        Args:
            x: Input tensor, shape [..., D].

        Returns:
            Encoded tensor, shape [..., D·(1+2L)] or [..., D·2L].
        """
        parts: list[Tensor] = []
        if self.include_input:
            parts.append(x)

        # x: [..., D], freqs: [L]
        # Broadcast: x[..., None, :] * freqs[..., None] → [..., L, D]
        # Then reshape to [..., L*D] per sin/cos block
        freqs = self.freqs.to(dtype=x.dtype, device=x.device)  # [L]
        scaled = x.unsqueeze(-2) * freqs.unsqueeze(-1)  # [..., L, D]
        # Flatten: [..., L*D]
        shape = scaled.shape[:-2] + (self.L * x.shape[-1],)

        sin_part = torch.sin(scaled).reshape(shape)  # [..., L*D]
        cos_part = torch.cos(scaled).reshape(shape)  # [..., L*D]

        # Interleave: for each frequency band, sin block then cos block
        # Layout: [sin(f0·p), cos(f0·p), sin(f1·p), cos(f1·p), ...]
        # Each block is D-dimensional
        D = x.shape[-1]
        interleaved_parts: list[Tensor] = []
        for l in range(self.L):
            interleaved_parts.append(sin_part[..., l * D:(l + 1) * D])
            interleaved_parts.append(cos_part[..., l * D:(l + 1) * D])

        parts.extend(interleaved_parts)
        return torch.cat(parts, dim=-1)

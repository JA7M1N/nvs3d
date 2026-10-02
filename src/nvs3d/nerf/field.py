"""NeRF MLP field: maps (position, direction) → (density, color).

(verify): 8 layers × 256, ReLU, skip concat of γ(x) at layer 5 — blueprint line 155.
(verify): σ head (1) + 256-d feature; concat γ(d) → 128 → RGB (3, sigmoid) — line 155.
(verify): σ = softplus(raw + b) — blueprint line 156.
(verify): Density bias init: "bias so initial σ is small" — line 157. No exact value.

skip_layer=5: The input to layer 5 (0-indexed) is concatenated with γ(x).
Layers 0–4 are standard; layer 5 sees [output_of_layer_4 || γ(x)].

Architecture:
    Backbone (density path):
        layer_0: Linear(pe_pos_dim, 256) + ReLU
        layers 1-4: Linear(256, 256) + ReLU
        layer_5: Linear(256 + pe_pos_dim, 256) + ReLU  ← skip concat
        layers 6-7: Linear(256, 256) + ReLU
    σ head: Linear(256, 1), then softplus
    Color path:
        color_hidden: Linear(256 + pe_dir_dim, 128) + ReLU
        color_out: Linear(128, 3) + sigmoid
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from nvs3d.nerf.encoding import PositionalEncoding


class NeRFField(nn.Module):
    """NeRF MLP: maps (position, view direction) → (density σ, color RGB).

    Args:
        pos_L: Frequency bands for position PE (default 10 → dim 63).
        dir_L: Frequency bands for direction PE (default 4 → dim 27).
        hidden: Hidden layer width (default 256).
        skip_layer: Layer index (0-based) where γ(x) is concatenated (default 5).
        num_layers: Number of backbone layers (default 8).
        density_bias: Bias added to raw density before softplus.
            (verify): Blueprint says "bias so initial σ is small (avoid opaque
            start)" but does not give an exact value.  We use -1.0 so that
            softplus(-1 + 0) ≈ 0.31 (small density at initialization).
    """

    def __init__(
        self,
        pos_L: int = 10,
        dir_L: int = 4,
        hidden: int = 256,
        skip_layer: int = 5,
        num_layers: int = 8,
        density_bias: float = -1.0,
    ) -> None:
        super().__init__()
        self.skip_layer = skip_layer
        self.density_bias = density_bias

        # Positional encodings
        self.pe_pos = PositionalEncoding(L=pos_L, include_input=True)
        self.pe_dir = PositionalEncoding(L=dir_L, include_input=True)
        pe_pos_dim = self.pe_pos.output_dim(3)  # 63 for L=10
        pe_dir_dim = self.pe_dir.output_dim(3)  # 27 for L=4
        self._pe_pos_dim = pe_pos_dim

        # Backbone layers (stored as nn.ModuleList for named access)
        backbone = nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                in_dim = pe_pos_dim
            elif i == skip_layer:
                in_dim = hidden + pe_pos_dim  # skip concat
            else:
                in_dim = hidden
            backbone.append(nn.Linear(in_dim, hidden))
        self.backbone = backbone

        # σ head
        self.sigma_head = nn.Linear(hidden, 1)

        # Color path
        self.color_hidden = nn.Linear(hidden + pe_dir_dim, 128)
        self.color_out = nn.Linear(128, 3)

        # Initialize
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights: Xavier uniform for all layers.

        Density head bias is set so initial σ is small.
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # Set density head bias so softplus(bias + raw) is small at init
        # raw is ~0 at init (Xavier with zero bias), so softplus(density_bias)
        # should be small.
        nn.init.constant_(self.sigma_head.bias, self.density_bias)

    def forward(self, pos: Tensor, dirs: Tensor) -> tuple[Tensor, Tensor]:
        """Evaluate the NeRF field.

        Args:
            pos: Positions, shape [B, 3].
            dirs: View directions, shape [B, 3].

        Returns:
            sigma: Density, shape [B, 1]. Non-negative (softplus).
            rgb: Color, shape [B, 3]. In (0, 1) (sigmoid).
        """
        # Encode inputs
        gamma_x = self.pe_pos(pos)   # [B, pe_pos_dim]
        gamma_d = self.pe_dir(dirs)  # [B, pe_dir_dim]

        # Backbone
        h = gamma_x
        for i, layer in enumerate(self.backbone):
            if i == self.skip_layer:
                h = torch.cat([h, gamma_x], dim=-1)
            h = F.relu(layer(h))

        # Density (depends only on position)
        sigma = F.softplus(self.sigma_head(h) + self.density_bias)  # [B, 1]

        # Color (depends on position + direction)
        feature = h  # 256-d backbone output
        color_input = torch.cat([feature, gamma_d], dim=-1)  # [B, 256+27]
        color = F.relu(self.color_hidden(color_input))  # [B, 128]
        rgb = torch.sigmoid(self.color_out(color))  # [B, 3]

        return sigma, rgb

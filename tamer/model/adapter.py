"""Small identity-initialized adapters for controlled domain adaptation."""

import torch
import torch.nn as nn
from torch import Tensor


class GatedBottleneckAdapter(nn.Module):
    """Residual bottleneck adapter with a learned scalar gate.

    The up projection is zero-initialized, so enabling an adapter preserves the
    pretrained function exactly at step zero while still allowing gradients to
    update the new branch.
    """

    def __init__(
        self,
        d_model: int,
        bottleneck_dim: int = 64,
        dropout: float = 0.1,
        gate_init_bias: float = -2.0,
    ) -> None:
        super().__init__()
        if bottleneck_dim <= 0:
            raise ValueError("bottleneck_dim must be positive")
        self.norm = nn.LayerNorm(d_model)
        self.down = nn.Linear(d_model, bottleneck_dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.up = nn.Linear(bottleneck_dim, d_model)
        self.gate_logit = nn.Parameter(torch.tensor(float(gate_init_bias)))
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    @property
    def gate(self) -> Tensor:
        return torch.sigmoid(self.gate_logit)

    def forward(self, x: Tensor) -> Tensor:
        update = self.up(self.dropout(self.activation(self.down(self.norm(x)))))
        return x + self.gate * update

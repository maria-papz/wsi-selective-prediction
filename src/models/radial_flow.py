import math
from typing import Tuple

import torch
import torch.nn.functional as F
from torch import nn

from src.models.flow_base import NormalizingFlow, Transform


class RadialTransform(Transform):
    """
    Radial transformation (Rezende & Mohamed, 2015).
    Applies radial contractions/expansions around a learned reference point.
    Directly from PICTURE's NatPN implementation.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.reference    = nn.Parameter(torch.empty(dim))
        self.alpha_prime  = nn.Parameter(torch.empty(1))
        self.beta_prime   = nn.Parameter(torch.empty(1))
        self.reset_parameters()

    def reset_parameters(self):
        std = 1 / math.sqrt(self.reference.size(0))
        nn.init.uniform_(self.reference,   -std, std)
        nn.init.uniform_(self.alpha_prime, -std, std)
        nn.init.uniform_(self.beta_prime,  -std, std)

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        dim   = self.reference.size(0)
        alpha = F.softplus(self.alpha_prime)              # [1]
        beta  = -alpha + F.softplus(self.beta_prime)      # [1]

        diff    = z - self.reference                      # [*, D]
        r       = diff.norm(dim=-1, keepdim=True)         # [*, 1]
        h       = (alpha + r).reciprocal()                # [*, 1]
        beta_h  = beta * h                                # [*, 1]
        y       = z + beta_h * diff                       # [*, D]

        h_d         = -(h ** 2)                                       # [*, 1]
        log_det_lhs = (dim - 1) * beta_h.log1p()                     # [*, 1]
        log_det_rhs = (beta_h + beta * h_d * r).log1p()              # [*, 1]
        log_det     = (log_det_lhs + log_det_rhs).squeeze(-1)        # [*]

        return y, log_det


class RadialFlow(NormalizingFlow):
    """
    Normalizing flow consisting of a series of radial transforms.
    Used by PICTURE's NatPN for epistemic uncertainty estimation.

    Args:
        dim:        input dimension (= ABMIL latent_dim)
        num_layers: number of sequential radial transforms (default 6,
                    matching PICTURE's config)
    """

    def __init__(self, dim: int, num_layers: int = 6):
        transforms = [RadialTransform(dim) for _ in range(num_layers)]
        super().__init__(transforms)
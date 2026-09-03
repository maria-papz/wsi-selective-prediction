import math
from typing import List, TypeVar
import torch
from torch import nn


class Transform(nn.Module):
    """Base class for normalizing flow transforms."""
    def forward(self, z: torch.Tensor):
        raise NotImplementedError


class NormalizingFlow(nn.Module):
    """
    Computes log p(z) under a standard Normal after applying a sequence
    of invertible transforms. Each transform returns (z_new, log_det).
    """

    def __init__(self, transforms: List[Transform]):
        super().__init__()
        self.transforms = nn.ModuleList(transforms)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: [*, dim]
        Returns:
            log_prob: [*]
        """
        batch_size = z.size()[:-1]
        dim = z.size(-1)
        log_det_sum = z.new_zeros(batch_size)

        for transform in self.transforms:
            z, log_det = transform.forward(z)
            log_det_sum += log_det

        const = dim * math.log(2 * math.pi)
        norm = torch.einsum("...ij,...ij->...i", z, z)
        normal_log_prob = -0.5 * (const + norm)
        return normal_log_prob + log_det_sum
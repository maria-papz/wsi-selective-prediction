from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from src.models.flow_base import NormalizingFlow, Transform
from src.models.batch_norm_transform import BatchNormTransform


#  MaskedAutoregressiveTransform 
class MaskedAutoregressiveTransform(Transform):
    r"""
    Masked Autoregressive Transform (Papamakarios et al., 2018).
    """

    def __init__(self, dim: int, hidden_dims: List[int]):
        super().__init__()
        self.net = MADE(dim, hidden_dims)

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mean, logscale = self.net(z).chunk(2, dim=-1)
        logscale = logscale.tanh()
        out      = (z - mean) * torch.exp(-logscale)
        log_det  = -logscale.sum(-1)
        return out, log_det


#  MADE 
class MADE(nn.Sequential):
    """
    Masked Autoencoder for Distribution Estimation (Germain et al., 2015).
    """

    def __init__(self, input_dim: int, hidden_dims: List[int]):
        assert len(hidden_dims) > 0, "MADE must have at least one hidden layer."
        dims         = [input_dim] + hidden_dims + [input_dim * 2]
        hidden_masks = _create_masks(input_dim, hidden_dims)
        layers       = []
        for i, (in_d, out_d) in enumerate(zip(dims, dims[1:])):
            if i > 0:
                layers.append(nn.LeakyReLU())
            layers.append(_MaskedLinear(in_d, out_d, mask=hidden_masks[i]))
        super().__init__(*layers)


class _MaskedLinear(nn.Linear):
    mask: torch.Tensor

    def __init__(self, in_features: int, out_features: int, mask: torch.Tensor):
        super().__init__(in_features, out_features)
        self.register_buffer("mask", mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight * self.mask, self.bias)

    def __repr__(self):
        return (f"MaskedLinear(in_features={self.in_features}, "
                f"out_features={self.out_features})")


def _create_masks(input_dim: int, hidden_dims: List[int]) -> List[torch.Tensor]:
    permutation    = torch.randperm(input_dim)
    input_degrees  = permutation + 1
    hidden_degrees = [_sample_degrees(1, input_dim - 1, d) for d in hidden_dims]
    output_degrees = permutation.repeat(2)
    all_degrees    = [input_degrees] + hidden_degrees + [output_degrees]
    return [
        _create_single_mask(in_deg, out_deg)
        for in_deg, out_deg in zip(all_degrees, all_degrees[1:])
    ]


def _create_single_mask(in_degrees: torch.Tensor,
                         out_degrees: torch.Tensor) -> torch.Tensor:
    return (out_degrees.unsqueeze(-1) >= in_degrees).float()


def _sample_degrees(minimum: int, maximum: int, num: int) -> torch.Tensor:
    return torch.linspace(minimum, maximum, steps=num).round()


#  MaskedAutoregressiveFlow 
class MaskedAutoregressiveFlow(NormalizingFlow):
    """
    Normalizing flow using masked autoregressive transforms with optional
    BatchNorm between layers for stability.

    Args:
        dim:              input dimension
        num_layers:       number of MAF transforms
        num_hidden_layers: hidden layers per MADE network
        hidden_layer_size: hidden dim (default: dim*3+1)
        use_batch_norm:   insert BatchNorm between transforms
    """

    def __init__(
        self,
        dim: int,
        num_layers: int = 4,
        num_hidden_layers: int = 1,
        hidden_layer_size: Optional[int] = None,
        use_batch_norm: bool = True,
    ):
        transforms = []
        for i in range(num_layers):
            if i > 0 and use_batch_norm:
                transforms.append(BatchNormTransform(dim))
            transforms.append(
                MaskedAutoregressiveTransform(
                    dim,
                    [hidden_layer_size or (dim * 3 + 1)] * num_hidden_layers,
                )
            )
        super().__init__(transforms)
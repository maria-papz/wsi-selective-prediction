import torch
import torch.nn as nn
import torch.nn.functional as F


class GatedAttention(nn.Module):
    """Gated attention pooling (Ilse et al. 2018).
    a_i = softmax( w^T (tanh(V h_i) * sigmoid(U h_i)) )
    """

    def __init__(self, in_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.V = nn.Linear(in_dim, hidden_dim)
        self.U = nn.Linear(in_dim, hidden_dim)
        self.w = nn.Linear(hidden_dim, 1)

    def forward(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # h: [N, in_dim]
        a_v = torch.tanh(self.V(h))       # [N, hidden]
        a_u = torch.sigmoid(self.U(h))    # [N, hidden]
        scores = self.w(a_v * a_u)        # [N, 1]
        attn = F.softmax(scores, dim=0)   # [N, 1]
        z = (attn * h).sum(dim=0)         # [in_dim]
        return z, attn.squeeze(-1)


class ABMIL(nn.Module):
    """
    Attention-Based MIL aggregator.

    Patch features -> projected embeddings -> gated attention pooling
    -> slide embedding z -> class logits.

    """

    def __init__(self, in_dim: int, latent_dim: int = 256,
                 num_classes: int = 2, dropout: float = 0.25):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, latent_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.attention = GatedAttention(latent_dim, hidden_dim=latent_dim)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(latent_dim, num_classes),
        )

    def forward(self, h: torch.Tensor,
                return_attention: bool = False
                ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            h: patch features [N, in_dim]
            return_attention: if True, also return attention weights
        Returns:
            logits: [num_classes]
            z: slide embedding [latent_dim]
            attn (optional): attention weights [N]
        """
        h = self.proj(h)                        # [N, latent_dim]
        z, attn = self.attention(h)             # z: [latent_dim]
        logits = self.classifier(z)             # [num_classes]
        if return_attention:
            return logits, z, attn
        return logits, z
from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Literal, NamedTuple, Sequence, Tuple

import torch
from torch import nn


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------
def clamp_preserve_gradients(
    x: torch.Tensor,
    lower: float,
    upper: float,
) -> torch.Tensor:
    return x + (x.clamp(min=lower, max=upper) - x).detach()


# -----------------------------------------------------------------------------
# Original NatPN distribution components retained for compatibility
# -----------------------------------------------------------------------------
class PosteriorUpdate(NamedTuple):
    sufficient_statistics: torch.Tensor
    log_evidence: torch.Tensor


class Likelihood(ABC):
    @abstractmethod
    def mean(self) -> torch.Tensor:
        ...

    @abstractmethod
    def uncertainty(self) -> torch.Tensor:
        ...

    @abstractmethod
    def expected_sufficient_statistics(self) -> torch.Tensor:
        ...


class Posterior(ABC):
    @abstractmethod
    def expected_log_likelihood(self, data: torch.Tensor) -> torch.Tensor:
        ...

    @abstractmethod
    def entropy(self) -> torch.Tensor:
        ...

    @abstractmethod
    def maximum_a_posteriori(self) -> Likelihood:
        ...


class Categorical(Likelihood):
    def __init__(self, logits: torch.Tensor):
        self.logits = logits

    def mean(self) -> torch.Tensor:
        return self.logits.argmax(dim=-1)

    def uncertainty(self) -> torch.Tensor:
        return -(self.logits * self.logits.exp()).sum(dim=-1)

    def expected_sufficient_statistics(self) -> torch.Tensor:
        return self.logits.exp()


class Dirichlet(Posterior):
    def __init__(self, alpha: torch.Tensor):
        self.alpha = alpha

    def expected_log_likelihood(self, data: torch.Tensor) -> torch.Tensor:
        a0 = self.alpha.sum(dim=-1)
        a_true = self.alpha.gather(-1, data.unsqueeze(-1)).squeeze(-1)
        return a_true.digamma() - a0.digamma()

    def entropy(self) -> torch.Tensor:
        k = self.alpha.size(-1)
        a0 = self.alpha.sum(dim=-1)

        approx = (
            0.5 * (k - 1) * (1 + math.log(2 * math.pi))
            + 0.5 * self.alpha.log().sum(dim=-1)
            - (k - 0.5) * a0.log()
        )

        exact = (
            self.alpha.lgamma().sum(dim=-1)
            - a0.lgamma()
            - (k - a0) * a0.digamma()
            - ((self.alpha - 1) * self.alpha.digamma()).sum(dim=-1)
        )

        return torch.where(a0 >= 10000, approx, exact)

    def maximum_a_posteriori(self) -> Categorical:
        return Categorical(
            self.alpha.log() - self.alpha.sum(dim=-1, keepdim=True).log()
        )


class DirichletPrior(nn.Module):
    sufficient_statistics: torch.Tensor
    evidence: torch.Tensor

    def __init__(self, num_categories: int, evidence: float):
        super().__init__()
        self.register_buffer(
            "sufficient_statistics",
            torch.ones(num_categories) / num_categories,
        )
        self.register_buffer("evidence", torch.as_tensor(float(evidence)))

    def update(self, update: PosteriorUpdate) -> Dirichlet:
        update_alpha = (
            update.sufficient_statistics
            * update.log_evidence.exp().unsqueeze(-1)
        )
        return Dirichlet(
            update_alpha + self.sufficient_statistics * self.evidence
        )


CertaintyBudget = Literal["constant", "exp-half", "exp", "normal"]


class EvidenceScaler(nn.Module):
    def __init__(
        self,
        dim: int,
        budget: CertaintyBudget,
        clamp: bool = True,
    ):
        super().__init__()

        if budget == "exp-half":
            self.log_scale = 0.5 * dim
        elif budget == "exp":
            self.log_scale = dim
        elif budget == "normal":
            self.log_scale = 0.5 * math.log(4 * math.pi) * dim
        else:
            self.log_scale = 0.0

        self.clamp = clamp

    def forward(self, log_evidence: torch.Tensor) -> torch.Tensor:
        scaled = log_evidence + self.log_scale
        if self.clamp:
            return clamp_preserve_gradients(scaled, -30.0, 30.0)
        return scaled


class CategoricalOutput(nn.Module):
    def __init__(self, dim: int, num_classes: int):
        super().__init__()
        self.linear = nn.Linear(dim, num_classes)
        self.prior = DirichletPrior(
            num_categories=num_classes,
            evidence=num_classes,
        )

    def forward(self, x: torch.Tensor) -> Categorical:
        return Categorical(self.linear(x).log_softmax(dim=-1))


class NaturalPosteriorNetworkModel(nn.Module):
    """Original single-flow NatPN retained for compatibility."""

    def __init__(
        self,
        latent_dim: int,
        encoder: nn.Module,
        flow: nn.Module,
        output: CategoricalOutput,
        clamp_scaling: bool = True,
        certainty_budget: CertaintyBudget = "normal",
        locked_encoder: bool = False,
    ):
        super().__init__()
        self.encoder = encoder
        self.flow = flow
        self.output = output
        self.scaler = EvidenceScaler(
            latent_dim,
            certainty_budget,
            clamp=clamp_scaling,
        )
        self.locked_encoder = locked_encoder
        self.latent_dim = latent_dim

    def forward(self, x: torch.Tensor) -> Tuple[Dirichlet, torch.Tensor]:
        update, log_prob = self.posterior_update(x)
        return self.output.prior.update(update), log_prob

    def posterior_update(
        self,
        x: torch.Tensor,
    ) -> Tuple[PosteriorUpdate, torch.Tensor]:
        if self.locked_encoder:
            with torch.no_grad():
                z = self.encoder(x)
        else:
            z = self.encoder(x)

        prediction = self.output(z)
        sufficient_statistics = prediction.expected_sufficient_statistics()
        log_prob = self.flow(z)
        log_evidence = self.scaler(log_prob)

        return PosteriorUpdate(sufficient_statistics, log_evidence), log_prob

    def calibrate_epistemic(self, train_log_probs: torch.Tensor) -> None:
        self.register_buffer("ep_mean", train_log_probs.mean())
        self.register_buffer("ep_std", train_log_probs.std().clamp(min=1.0))

    def epistemic_confidence_from_log_prob(
        self,
        log_prob: torch.Tensor,
    ) -> torch.Tensor:
        if hasattr(self, "ep_mean") and hasattr(self, "ep_std"):
            z = (log_prob - self.ep_mean) / self.ep_std
            return torch.sigmoid(z)

        baseline = -0.5 * self.latent_dim * math.log(2 * math.pi)
        return torch.sigmoid(log_prob - baseline)

    def epistemic_confidence(self, x: torch.Tensor) -> torch.Tensor:
        _, log_prob = self.forward(x)
        return self.epistemic_confidence_from_log_prob(log_prob)

    def ood_score_from_log_prob(self, log_prob: torch.Tensor) -> torch.Tensor:
        return -log_prob

    def aleatoric_confidence(self, x: torch.Tensor) -> torch.Tensor:
        posterior, _ = self.forward(x)
        entropy = posterior.maximum_a_posteriori().uncertainty()
        return torch.sigmoid(-entropy)


class BayesianLoss(nn.Module):
    def __init__(self, entropy_weight: float = 1e-5):
        super().__init__()
        self.entropy_weight = entropy_weight

    def forward(self, y_pred: Dirichlet, y_true: torch.Tensor) -> torch.Tensor:
        nll = -y_pred.expected_log_likelihood(y_true)
        loss = nll - self.entropy_weight * y_pred.entropy()
        return loss.mean()


class PatchDistributionLoss(nn.Module):
    """Original single-flow combined loss retained for compatibility."""

    def __init__(
        self,
        entropy_weight: float = 1e-5,
        density_weight: float = 1e-4,
    ):
        super().__init__()
        self.bayesian_loss = BayesianLoss(entropy_weight=entropy_weight)
        self.density_weight = density_weight

    def forward(
        self,
        posterior: Dirichlet,
        y_true: torch.Tensor,
        log_prob: torch.Tensor,
    ):
        cls_loss = self.bayesian_loss(posterior, y_true)
        density_loss = -log_prob.mean()
        weighted_density_loss = self.density_weight * density_loss
        loss = cls_loss + weighted_density_loss
        return (
            loss,
            cls_loss.detach(),
            density_loss.detach(),
            weighted_density_loss.detach(),
        )


class GaussianDensity(nn.Module):
    """
    Closed-form (unconditional) multivariate Gaussian density, as a drop-in
    replacement for a normalizing flow in NaturalPosteriorNetworkModel --
    same forward(z) -> log_prob contract as NormalizingFlow, fit via
    method-of-moments (no gradient-based training, no flow-training
    instability) rather than maximum likelihood via backprop.

    Ablation partner for RadialFlow/MaskedAutoregressiveFlow: is a flow's
    extra flexibility worth it at this sample size, or does a closed-form
    Gaussian (identical in spirit to the Mahalanobis ablation already used
    for the patch-level OOD detector) do just as well grounding NatPN's
    evidence term.
    """

    def __init__(self, dim: int, eps: float = 1e-3):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("precision", torch.eye(dim))
        self.register_buffer("log_det_cov", torch.zeros(()))
        self.register_buffer("fitted", torch.tensor(False))

    @torch.no_grad()
    def fit(self, z: torch.Tensor) -> None:
        """Method-of-moments fit. z: [N, dim]. Diagonal-loaded covariance,
        same shrinkage convention as the existing Mahalanobis ablation
        (cov + eps * I) rather than a new regularization scheme."""
        mean = z.mean(dim=0)
        centered = z - mean
        cov = (centered.T @ centered) / z.size(0)
        cov = cov + self.eps * torch.eye(self.dim, device=z.device)

        self.mean.copy_(mean)
        self.precision.copy_(torch.linalg.inv(cov))
        sign, logdet = torch.linalg.slogdet(cov)
        self.log_det_cov.copy_(logdet)
        self.fitted.fill_(True)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if not bool(self.fitted.item()):
            raise RuntimeError("GaussianDensity.fit(z) must be called before forward().")
        diff = z - self.mean
        mahalanobis = torch.einsum("...i,ij,...j->...", diff, self.precision, diff)
        const = self.dim * math.log(2 * math.pi)
        return -0.5 * (const + self.log_det_cov + mahalanobis)


# -----------------------------------------------------------------------------
# Class-conditional patch-density model
# -----------------------------------------------------------------------------
class OrthogonalProjection(nn.Module):
    """
    Fixed or trainable orthogonal linear projection.

    For final density fitting, keep this fixed so the target latent distribution
    does not move while the flows are being optimized.
    """

    def __init__(
        self,
        in_dim: int,
        latent_dim: int,
        trainable: bool = False,
    ):
        super().__init__()

        if latent_dim > in_dim:
            raise ValueError(
                f"latent_dim ({latent_dim}) cannot exceed in_dim ({in_dim}) "
                "for this orthogonal projection."
            )

        self.linear = nn.Linear(in_dim, latent_dim, bias=False)
        nn.init.orthogonal_(self.linear.weight)
        self.linear.weight.requires_grad_(trainable)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class ClassConditionalPatchDensityModel(nn.Module):
    """
    One shared fixed projection plus one density flow per tumour class.

    The model returns a matrix of class-conditional log densities:

        class_log_probs[i, k] = log p(z_i | class=k)

    OOD calibration is performed independently for every class. A patch is
    globally OOD when it falls below the calibrated likelihood threshold for
    every known class.
    """

    def __init__(
        self,
        encoder: nn.Module,
        flows: Sequence[nn.Module],
        latent_dim: int,
        num_classes: int,
        class_names: Sequence[str] | None = None,
    ):
        super().__init__()

        if len(flows) != num_classes:
            raise ValueError(
                f"Expected {num_classes} flows, received {len(flows)}."
            )

        self.encoder = encoder
        self.flows = nn.ModuleList(flows)
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.class_names = tuple(
            class_names
            if class_names is not None
            else [f"class_{k}" for k in range(num_classes)]
        )

        self.register_buffer("latent_mean", torch.zeros(latent_dim))
        self.register_buffer("latent_std", torch.ones(latent_dim))
        self.register_buffer(
            "latent_normalizer_fitted",
            torch.tensor(False, dtype=torch.bool),
        )

        self.register_buffer(
            "class_log_prior",
            torch.full((num_classes,), -math.log(num_classes)),
        )

        self.register_buffer("class_lp_mean", torch.zeros(num_classes))
        self.register_buffer("class_lp_std", torch.ones(num_classes))
        self.register_buffer(
            "class_lp_threshold",
            torch.full((num_classes,), float("-inf")),
        )
        self.register_buffer(
            "ood_calibrated",
            torch.tensor(False, dtype=torch.bool),
        )

    def encode_raw(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encode_raw(x)
        if bool(self.latent_normalizer_fitted.item()):
            z = (z - self.latent_mean) / self.latent_std
        return z

    @torch.no_grad()
    def fit_latent_normalizer(self, raw_latents: torch.Tensor) -> None:
        if raw_latents.ndim != 2 or raw_latents.size(-1) != self.latent_dim:
            raise ValueError(
                "raw_latents must have shape [num_patches, latent_dim]."
            )

        mean = raw_latents.mean(dim=0)
        std = raw_latents.std(dim=0, unbiased=False).clamp(min=1e-4)

        self.latent_mean.copy_(mean.to(self.latent_mean.device))
        self.latent_std.copy_(std.to(self.latent_std.device))
        self.latent_normalizer_fitted.fill_(True)

    @torch.no_grad()
    def set_class_priors(self, class_counts: torch.Tensor) -> None:
        class_counts = class_counts.float().clamp(min=1.0)
        priors = class_counts / class_counts.sum()
        self.class_log_prior.copy_(priors.log().to(self.class_log_prior.device))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encode(x)
        class_log_probs = torch.stack(
            [flow(z) for flow in self.flows],
            dim=-1,
        )
        return class_log_probs

    def selected_log_prob(
        self,
        class_log_probs: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        return class_log_probs.gather(
            dim=1,
            index=labels.unsqueeze(1),
        ).squeeze(1)

    def density_class_log_posterior_from_log_probs(
        self,
        class_log_probs: torch.Tensor,
    ) -> torch.Tensor:
        joint = class_log_probs + self.class_log_prior.unsqueeze(0)
        return joint.log_softmax(dim=-1)

    def density_class_probabilities_from_log_probs(
        self,
        class_log_probs: torch.Tensor,
    ) -> torch.Tensor:
        return self.density_class_log_posterior_from_log_probs(
            class_log_probs
        ).exp()

    @torch.no_grad()
    def calibrate_ood_from_class_values(
        self,
        class_values: Sequence[torch.Tensor],
        quantile: float = 0.05,
    ) -> None:
        if not 0.0 < quantile < 0.5:
            raise ValueError("quantile must be between 0 and 0.5.")
        if len(class_values) != self.num_classes:
            raise ValueError(
                f"Expected values for {self.num_classes} classes, "
                f"received {len(class_values)}."
            )

        means = []
        stds = []
        thresholds = []

        for class_idx, values in enumerate(class_values):
            values = values.detach().float().flatten()
            if values.numel() < 2:
                raise ValueError(
                    f"Not enough calibration values for class {class_idx}."
                )

            means.append(values.mean())
            stds.append(values.std(unbiased=False).clamp(min=1e-4))
            thresholds.append(torch.quantile(values, quantile))

        means_t = torch.stack(means).to(self.class_lp_mean.device)
        stds_t = torch.stack(stds).to(self.class_lp_std.device)
        thresholds_t = torch.stack(thresholds).to(
            self.class_lp_threshold.device
        )

        self.class_lp_mean.copy_(means_t)
        self.class_lp_std.copy_(stds_t)
        self.class_lp_threshold.copy_(thresholds_t)
        self.ood_calibrated.fill_(True)

    def class_ood_scores_from_log_probs(
        self,
        class_log_probs: torch.Tensor,
    ) -> torch.Tensor:
        if not bool(self.ood_calibrated.item()):
            raise RuntimeError(
                "OOD calibration has not been fitted. Call "
                "calibrate_ood_from_class_values() first."
            )

        # Score is 0.5 exactly at the calibrated class threshold.
        standardized_margin = (
            self.class_lp_threshold.unsqueeze(0) - class_log_probs
        ) / self.class_lp_std.unsqueeze(0)
        return torch.sigmoid(standardized_margin)

    def class_ood_flags_from_log_probs(
        self,
        class_log_probs: torch.Tensor,
    ) -> torch.Tensor:
        if not bool(self.ood_calibrated.item()):
            raise RuntimeError(
                "OOD calibration has not been fitted. Call "
                "calibrate_ood_from_class_values() first."
            )
        return class_log_probs < self.class_lp_threshold.unsqueeze(0)

    def global_ood_score_from_log_probs(
        self,
        class_log_probs: torch.Tensor,
    ) -> torch.Tensor:
        class_scores = self.class_ood_scores_from_log_probs(class_log_probs)
        # A patch is ID if it resembles at least one known class.
        return class_scores.min(dim=-1).values

    def global_ood_flag_from_log_probs(
        self,
        class_log_probs: torch.Tensor,
    ) -> torch.Tensor:
        class_flags = self.class_ood_flags_from_log_probs(class_log_probs)
        return class_flags.all(dim=-1)

    def infer(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        class_log_probs = self.forward(x)
        class_probabilities = self.density_class_probabilities_from_log_probs(
            class_log_probs
        )

        result = {
            "class_log_probs": class_log_probs,
            "density_class_probabilities": class_probabilities,
        }

        if bool(self.ood_calibrated.item()):
            class_ood_scores = self.class_ood_scores_from_log_probs(
                class_log_probs
            )
            class_ood_flags = self.class_ood_flags_from_log_probs(
                class_log_probs
            )
            result.update(
                {
                    "class_ood_scores": class_ood_scores,
                    "class_ood_flags": class_ood_flags,
                    "global_ood_score": class_ood_scores.min(dim=-1).values,
                    "global_ood_flag": class_ood_flags.all(dim=-1),
                }
            )

        return result


class ClassConditionalDensityLoss(nn.Module):
    """
    Maximum-likelihood loss for separate class-conditional flows.

    Only log p(z | y) contributes for each patch. By default, the loss is
    averaged per class and then across classes present in the batch, preventing
    the majority tumour type from dominating optimization.
    """

    def __init__(self, balance_classes: bool = True):
        super().__init__()
        self.balance_classes = balance_classes

    def forward(
        self,
        class_log_probs: torch.Tensor,
        labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        selected_log_prob = class_log_probs.gather(
            dim=1,
            index=labels.unsqueeze(1),
        ).squeeze(1)

        if self.balance_classes:
            class_losses = []
            for class_idx in range(class_log_probs.size(1)):
                mask = labels == class_idx
                if mask.any():
                    class_losses.append(-selected_log_prob[mask].mean())

            if not class_losses:
                raise RuntimeError("No valid class labels were found in batch.")

            loss = torch.stack(class_losses).mean()
        else:
            loss = -selected_log_prob.mean()

        return loss, selected_log_prob.detach()

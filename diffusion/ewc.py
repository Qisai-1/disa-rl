"""
Elastic Weight Consolidation (EWC) for diffusion model online fine-tuning.

Problem it solves
-----------------
When the diffusion model is fine-tuned on online data during the offline→online
transition, it can forget the offline distribution it was carefully trained on.
This is catastrophic because the offline distribution provides the "prior
knowledge" about environment dynamics — the whole reason we trained it.

How EWC works
-------------
After offline training, we compute the Fisher information matrix F for each
parameter θ_i.  F_i estimates how important θ_i is for generating trajectories
from the offline distribution — specifically, it is the expected squared
gradient of the log-likelihood with respect to θ_i.

During online fine-tuning, we add a regularisation term to the loss:

    L_ewc = (λ_ewc / 2) · Σ_i  F_i · (θ_i − θ*_i)²

where θ*_i are the offline-trained parameter values.  This penalises moving
parameters that were important for the offline distribution.

EWC vs replay
-------------
An alternative to EWC is to replay offline data during fine-tuning (mix offline
and online samples in every batch).  We implement both:
  - EWC: zero storage overhead, approximates the constraint analytically
  - Replay buffer: exact but requires storing offline samples

For this codebase, EWC is the primary mechanism and replay is optional.

Implementation notes
--------------------
Computing the exact Fisher (over all parameters and the full offline dataset)
is expensive.  We use the *empirical* Fisher — estimated from one pass over a
representative subset of offline data — which is the standard approximation
used in the original EWC paper (Kirkpatrick et al., 2017).

The Fisher is computed from the *flow matching loss gradient* rather than a
log-likelihood (since our model is a flow matching model, not a density model).
The squared gradient of the flow matching loss w.r.t. each parameter serves as
a proxy for the Fisher information about trajectory generation quality.
"""

from __future__ import annotations
import sys, os as _os
_d = _os.path.dirname(_os.path.abspath(__file__))
if _d not in sys.path: sys.path.insert(0, _d)
if _os.path.dirname(_d) not in sys.path: sys.path.insert(0, _os.path.dirname(_d))
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Optional
from tqdm import tqdm


class EWC:
    """
    Elastic Weight Consolidation regulariser.

    Usage
    -----
    # 1. After offline training, instantiate EWC with the trained model
    #    and a representative offline data loader.
    ewc = EWC(model, cfm, offline_loader, device, n_batches=50)

    # 2. During online fine-tuning, add the penalty to the loss.
    ft_loss = cfm_loss + ewc.penalty(model)
    """

    def __init__(
        self,
        model:          nn.Module,
        cfm,                          # ConditionalFlowMatching instance
        data_loader:    DataLoader,
        device:         torch.device,
        n_batches:      int   = 50,   # batches to use for Fisher estimation
        lambda_ewc:     float = 500.0,
    ):
        self.device     = device
        self.lambda_ewc = lambda_ewc

        print(f"Computing EWC Fisher information ({n_batches} batches)...")

        # Store offline parameter values θ*
        self.params_star: Dict[str, torch.Tensor] = {
            name: param.data.clone().detach()
            for name, param in model.named_parameters()
            if param.requires_grad
        }

        # Compute empirical Fisher
        self.fisher: Dict[str, torch.Tensor] = self._compute_fisher(
            model, cfm, data_loader, n_batches
        )

        print(f"EWC ready.  Protecting {len(self.fisher)} parameter tensors.")

    def _compute_fisher(
        self,
        model:       nn.Module,
        cfm,
        loader:      DataLoader,
        n_batches:   int,
    ) -> Dict[str, torch.Tensor]:
        """
        Estimate Fisher information via squared gradients of the flow
        matching loss over a subset of the offline dataset.

        F_i ≈ E[(∂L/∂θ_i)²]

        We accumulate (∂L/∂θ_i)² over n_batches and normalise by count.
        """
        model.train()

        # Initialise accumulators to zero
        fisher = {
            name: torch.zeros_like(param)
            for name, param in model.named_parameters()
            if param.requires_grad
        }

        count = 0
        for i, batch in enumerate(tqdm(loader, total=n_batches, desc="Fisher estimation")):
            if i >= n_batches:
                break

            x1   = batch["trajectory"].to(self.device)
            cond = batch["condition"].to(self.device)

            model.zero_grad()
            loss, _ = cfm.loss(x1, cond)
            loss.backward()

            for name, param in model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    fisher[name] += param.grad.data.pow(2)

            count += 1

        # Normalise
        for name in fisher:
            fisher[name] /= max(count, 1)

        model.zero_grad()
        return fisher

    def penalty(self, model: nn.Module) -> torch.Tensor:
        """
        EWC regularisation loss.

        Returns scalar:   (λ_ewc / 2) · Σ_i  F_i · (θ_i − θ*_i)²
        """
        loss = torch.tensor(0.0, device=self.device)
        for name, param in model.named_parameters():
            if name in self.fisher:
                diff   = param - self.params_star[name].to(param.device)
                loss  += (self.fisher[name].to(param.device) * diff.pow(2)).sum()
        return (self.lambda_ewc / 2.0) * loss

    def update_reference(self, model: nn.Module) -> None:
        """
        Update the reference parameters θ* to the current model weights.

        Call this after each fine-tuning phase to "accept" the new weights
        as the new baseline.  This implements online EWC (progressive nets).
        """
        self.params_star = {
            name: param.data.clone().detach()
            for name, param in model.named_parameters()
            if param.requires_grad
        }

    def save(self, path: str) -> None:
        """Serialise Fisher + reference params for resuming later."""
        torch.save({
            "fisher":      {k: v.cpu() for k, v in self.fisher.items()},
            "params_star": {k: v.cpu() for k, v in self.params_star.items()},
            "lambda_ewc":  self.lambda_ewc,
        }, path)
        print(f"EWC state saved → {path}")

    @classmethod
    def load(cls, path: str, device: torch.device) -> "EWC":
        """Restore EWC state from a file (skip re-computing Fisher)."""
        ckpt = torch.load(path, map_location=device, weights_only=False)
        obj = cls.__new__(cls)
        obj.device     = device
        obj.lambda_ewc = ckpt["lambda_ewc"]
        obj.fisher     = {k: v.to(device) for k, v in ckpt["fisher"].items()}
        obj.params_star= {k: v.to(device) for k, v in ckpt["params_star"].items()}
        print(f"EWC state loaded from {path}")
        return obj

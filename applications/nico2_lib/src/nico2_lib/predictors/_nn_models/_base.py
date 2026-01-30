from dataclasses import dataclass
from typing import Any, List, Tuple

import lightning as L
import torch
from torch import nn
from torch.nn import functional as F


def vae_loss_function(
    target: torch.Tensor, pred: torch.Tensor, mu_z: torch.Tensor, logvar_z: torch.Tensor
) -> torch.Tensor:
    """MSE reconstruction + KL divergence."""
    recon_loss = F.mse_loss(pred, target, reduction="sum")
    kl_loss = -0.5 * torch.sum(1 + logvar_z - mu_z.pow(2) - logvar_z.exp())
    return recon_loss + kl_loss


class BaseVAE(L.LightningModule):
    """Base VAE class for predicting output_features from input_features."""

    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        lr: float,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.encoder = encoder
        self.decoder = decoder
        self.lr = lr

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass.
        If `log_space=True`, expects log1p-transformed inputs.
        If `log_space=False`, expects raw counts and applies log1p inside.
        """
        z, mu_z, logvar_z = self.encoder(x)
        mu_x = self.decoder(z)
        return mu_x, mu_z, logvar_z

    def training_step(self, batch: List[torch.Tensor], batch_idx: int) -> torch.Tensor:
        x = batch[0]
        mu_x, mu_z, logvar_z = self.forward(x)
        loss = vae_loss_function(x, mu_x, mu_z, logvar_z)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        """Compute validation loss in log space (same as training)."""
        mu_x, mu_z, logvar_z = self.forward(batch)
        loss = vae_loss_function(batch, mu_x, mu_z, logvar_z)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        return loss

    def predict_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        """Return model outputs transformed back to raw count space."""
        mu_x, _, _ = self.forward(batch)
        return mu_x

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.AdamW(self.parameters(), lr=self.lr)

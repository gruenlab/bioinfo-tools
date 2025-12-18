from typing import Tuple
import torch
from torch import nn
from torch.nn import functional as F
import lightning as L


def vae_loss_function(
    target: torch.Tensor, pred: torch.Tensor, mu_z: torch.Tensor, logvar_z: torch.Tensor
) -> torch.Tensor:
    """MSE reconstruction + KL divergence."""
    recon_loss = F.mse_loss(pred, target, reduction="sum")
    kl_loss = -0.5 * torch.sum(1 + logvar_z - mu_z.pow(2) - logvar_z.exp())
    return recon_loss + kl_loss


class BaseVAE(L.LightningModule):
    """Base VAE class for predicting output_features from input_features."""

    encoder: nn.Module
    decoder: nn.Module

    def __init__(
        self,
        input_features: int,
        output_features: int,
        latent_features: int,
        lr: float = 1e-4,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr

    def forward(
        self, x: torch.Tensor, log_space: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass.
        If `log_space=True`, expects log1p-transformed inputs.
        If `log_space=False`, expects raw counts and applies log1p inside.
        """
        if not log_space:
            x = torch.log1p(x)

        z, mu_z, logvar_z = self.encoder(x)
        mu_x = self.decoder(z)

        return mu_x, mu_z, logvar_z

    def training_step(self, batch, batch_idx):
        input_data, target_data = batch

        # operate entirely in log1p space
        input_log = torch.log1p(input_data)
        target_log = torch.log1p(target_data)

        mu_x, mu_z, logvar_z = self(input_log, log_space=True)

        loss = vae_loss_function(target_log, mu_x, mu_z, logvar_z)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """Compute validation loss in log space (same as training)."""
        input_data, target_data = batch
        input_log = torch.log1p(input_data)
        target_log = torch.log1p(target_data)

        mu_x, mu_z, logvar_z = self(input_log, log_space=True)
        loss = vae_loss_function(target_log, mu_x, mu_z, logvar_z)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        return loss

    def predict_step(self, batch, batch_idx):
        """Return model outputs transformed back to raw count space."""
        input_data, _ = batch  # we don’t need targets
        mu_x, _, _ = self(input_data, log_space=False)
        return torch.expm1(mu_x).clamp(min=0)  # ensure nonnegative counts

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)

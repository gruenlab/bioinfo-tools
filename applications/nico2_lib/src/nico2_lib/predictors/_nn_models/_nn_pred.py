from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Type

import lightning as L
import numpy as np
import torch
from numpy import number
from numpy.typing import NDArray
from torch.utils.data import DataLoader, TensorDataset

from nico2_lib.predictors._nn_models._models import LDVAE, LEVAE, LVAE, VAE, BaseVAE


@dataclass
class VaePredictor:
    vae_cls: Type[BaseVAE]

    # --- training ---
    batch_size: int = 64
    max_epochs: int = 200
    accelerator: str = "auto"
    devices: Optional[int] = None
    trainer_kwargs: Dict[str, Any] = field(default_factory=dict)

    # --- internal ---
    vae_kwargs: Dict[str, Any] = field(default_factory=dict)
    vae: Optional[BaseVAE] = field(init=False, default=None)
    _fit_trainer: Optional[L.Trainer] = field(init=False, default=None)
    _fitted: bool = field(init=False, default=False)

    def fit(self, X: NDArray[number], y: NDArray[number]) -> "VaePredictor":
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)

        input_features = X.shape[1]
        output_features = y.shape[1]

        self.vae = self.vae_cls(
            input_features=input_features,
            output_features=output_features,
            **self.vae_kwargs,
        )

        dataset = TensorDataset(
            torch.from_numpy(X),
            torch.from_numpy(y),
        )
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
        )

        self._fit_trainer = L.Trainer(
            max_epochs=self.max_epochs,
            accelerator=self.accelerator,
            devices=self.devices,
            enable_checkpointing=False,
            logger=False,
            **self.trainer_kwargs,
        )

        self._fit_trainer.fit(self.vae, train_dataloaders=loader)

        self._fitted = True
        return self

    def predict(self, X: NDArray[number]) -> NDArray[number]:
        if not self._fitted or self.vae is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        X = np.asarray(X, dtype=np.float32)
        x_tensor = torch.from_numpy(X)

        self.vae.eval()
        with torch.no_grad():
            mu_x, _, _ = self.vae(x_tensor, log_space=False)
            preds = torch.expm1(mu_x).clamp(min=0)

        return preds.cpu().numpy()


class LVAEPredictor(VaePredictor):
    def __init__(
        self,
        latent_features: int = 64,
        lr: float = 1e-3,
        batch_size: int = 64,
        max_epochs: int = 200,
        accelerator: str = "auto",
        devices: Optional[int] = None,
        trainer_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            vae_cls=LVAE,
            batch_size=batch_size,
            max_epochs=max_epochs,
            accelerator=accelerator,
            devices=devices,
            trainer_kwargs=trainer_kwargs or {},
        )
        self.vae_kwargs = {
            "latent_features": latent_features,
            "lr": lr,
        }


class LEVAEPredictor(VaePredictor):
    def __init__(
        self,
        latent_features: int,
        hidden_features_out: Optional[int] = None,
        lr: float = 1e-4,
        batch_size: int = 64,
        max_epochs: int = 200,
        accelerator: str = "auto",
        devices: Optional[int] = None,
        trainer_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            vae_cls=LEVAE,
            batch_size=batch_size,
            max_epochs=max_epochs,
            accelerator=accelerator,
            devices=devices,
            trainer_kwargs=trainer_kwargs or {},
        )
        self.vae_kwargs = {
            "latent_features": latent_features,
            "hidden_features_out": hidden_features_out,
            "lr": lr,
        }


class LDVAEPredictor(VaePredictor):
    def __init__(
        self,
        latent_features: int,
        hidden_features_in: Optional[int] = None,
        lr: float = 1e-4,
        batch_size: int = 64,
        max_epochs: int = 200,
        accelerator: str = "auto",
        devices: Optional[int] = None,
        trainer_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            vae_cls=LDVAE,
            batch_size=batch_size,
            max_epochs=max_epochs,
            accelerator=accelerator,
            devices=devices,
            trainer_kwargs=trainer_kwargs or {},
        )
        self.vae_kwargs = {
            "latent_features": latent_features,
            "hidden_features_in": hidden_features_in,
            "lr": lr,
        }


class VAEPredictor(VaePredictor):
    def __init__(
        self,
        latent_features: int,
        hidden_features_out: Optional[int] = None,
        hidden_features_in: Optional[int] = None,
        lr: float = 1e-4,
        batch_size: int = 64,
        max_epochs: int = 200,
        accelerator: str = "auto",
        devices: Optional[int] = None,
        trainer_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            vae_cls=VAE,
            batch_size=batch_size,
            max_epochs=max_epochs,
            accelerator=accelerator,
            devices=devices,
            trainer_kwargs=trainer_kwargs or {},
        )
        self.vae_kwargs = {
            "latent_features": latent_features,
            "hidden_features_out": hidden_features_out,
            "hidden_features_in": hidden_features_in,
            "lr": lr,
        }

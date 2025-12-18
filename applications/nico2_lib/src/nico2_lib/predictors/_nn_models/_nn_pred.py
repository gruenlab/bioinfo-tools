from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Type

import numpy as np
from numpy import number
from numpy.typing import NDArray
import torch
from torch.utils.data import DataLoader, TensorDataset
import lightning as L

from nico2_lib.predictors._nn_models._models import BaseVAE


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
            **self.vae_kwargs

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

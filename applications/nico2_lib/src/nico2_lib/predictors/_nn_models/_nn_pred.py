from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
from numpy import float32, number
from numpy.typing import NDArray
import torch
from torch.utils.data import DataLoader, TensorDataset
import lightning as L

from nico2_lib.predictors._nn_models._base import BaseVAE


@dataclass
class VaePredictor:
    vae: BaseVAE
    batch_size: int = 64
    max_epochs: int = 200
    accelerator: str = "auto"
    devices: Optional[int] = None
    trainer_kwargs: Optional[Dict[Any, Any]] = None

    def fit(self, X: NDArray[number], y: NDArray[number]):
        dataset = TensorDataset(
            torch.from_numpy(X, dtype=float32), torch.from_numpy(y, dtype=float32)
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
        if not self._fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        X = np.asarray(X, dtype=np.float32)
        x_tensor = torch.from_numpy(X)

        self.vae.eval()
        with torch.no_grad():
            mu_x, _, _ = self.vae(x_tensor, log_space=False)
            preds = torch.expm1(mu_x).clamp(min=0)

        return preds.cpu().numpy()

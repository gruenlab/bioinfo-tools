from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Tuple, Type

import lightning as L
import numpy as np
import torch
from numpy import clip, exp, intp, log1p, number
from numpy.typing import NDArray
from pandas._libs.tslibs.dtypes import NpyDatetimeUnit
from torch.utils.data import DataLoader, TensorDataset

from nico2_lib.predictors._nn_models._models import LDVAE, LEVAE, LVAE, VAE, BaseVAE

from ._nn import Decoder, LinearDecoder, VariationalEncoder, VariationalLinearEncoder


@dataclass
class BaseVaePredictor:
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
    _input_features: Optional[int] = field(init=False, default=None)
    _y_features: Optional[int] = field(init=False, default=None)

    def fit(self, X: NDArray[number], y: NDArray[number]) -> "BaseVaePredictor":
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)

        input_features = X.shape[1]
        y_features = y.shape[1]
        target = np.concatenate([X, y], axis=1)
        output_features = target.shape[1]

        self._input_features = input_features
        self._y_features = y_features

        self.vae = self.vae_cls(
            input_features=input_features,
            output_features=output_features,
            **self.vae_kwargs,
        )

        dataset = TensorDataset(
            torch.from_numpy(X),
            torch.from_numpy(target),
        )
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
        )

        self._fit_trainer = L.Trainer(
            max_epochs=self.max_epochs,
            accelerator=self.accelerator,
            devices=self.devices,  # pyright: ignore[reportArgumentType]
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

        if self._input_features is None or self._y_features is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        y_start = self._input_features
        y_end = y_start + self._y_features
        return preds[:, y_start:y_end].cpu().numpy()


class LVAEPredictor(BaseVaePredictor):
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


class LEVAEPredictor(BaseVaePredictor):
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


class LDVAEPredictor(BaseVaePredictor):
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


class VAEPredictor(BaseVaePredictor):
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


@dataclass
class VaePredictorN:
    latent_features: int
    hidden_features_out: int
    hidden_features_in: int
    lr: float = 1e-4
    transform: Tuple[
        Callable[[NDArray[number]], NDArray[number]],
        Callable[[NDArray[number]], NDArray[number]],
    ] = (lambda x: log1p(x), lambda x: exp(clip(x, min=0)))
    dataloader_kwargs: Dict[str, Any] = field(default_factory=dict)
    trainer_kwargs: Dict[str, Any] = field(default_factory=dict)

    _default_dataloader_kwargs: Dict[str, Any] = field(
        default_factory=lambda: {"batch_size": 64, "shuffle": True}
    )
    _default_trainer_kwargs: Dict[str, Any] = field(
        default_factory=lambda: {
            "max_epochs": 200,
            "enable_checkpointing": False,
            "logger": False,
        }
    )

    def _merged_dataloader_kwargs(self) -> Dict[str, Any]:
        return {**self._default_dataloader_kwargs, **self.dataloader_kwargs}

    def _merged_trainer_kwargs(self) -> Dict[str, Any]:
        return {**self._default_trainer_kwargs, **self.trainer_kwargs}

    def _build_encoder(self, n_features: int) -> VariationalEncoder:
        return VariationalEncoder(
            in_features=n_features,
            latent_features=self.latent_features,
            hidden_features=self.hidden_features_in,
        )

    def _build_decoder(self, n_features: int) -> Decoder:
        return Decoder(
            latent_features=self.latent_features,
            out_features=n_features,
            hidden_features=self.hidden_features_out,
        )

    def _fit_vae(
        self, encoder: VariationalEncoder, decoder: Decoder, X: NDArray[number]
    ) -> None:
        X = np.asarray(self.transform[0](X), dtype=np.float32)
        model = BaseVAE(encoder, decoder, lr=self.lr)
        dataset = TensorDataset(torch.from_numpy(X))
        loader = DataLoader(dataset, **self._merged_dataloader_kwargs())
        trainer = L.Trainer(**self._merged_trainer_kwargs())
        trainer.fit(model, train_dataloaders=loader)

    def _copy_decoder_for_query(self, indexer: NDArray[intp]) -> None:
        with torch.no_grad():
            self.decoder_query.hidden.weight.copy_(self.decoder_ref.hidden.weight)
            self.decoder_query.hidden.bias.copy_(self.decoder_ref.hidden.bias)
            self.decoder_query.mu_out.weight.copy_(
                self.decoder_ref.mu_out.weight[torch.from_numpy(indexer)]
            )
            self.decoder_query.mu_out.bias.copy_(
                self.decoder_ref.mu_out.bias[torch.from_numpy(indexer)]
            )

    def _freeze_decoder_query(self) -> None:
        for param in self.decoder_query.parameters():
            param.requires_grad = False

    def _predict_with_ref_decoder(self, X: NDArray[number]) -> NDArray[number]:
        predictor_model = BaseVAE(
            encoder=self.encoder_query, decoder=self.decoder_ref, lr=self.lr
        )
        X = np.asarray(self.transform[0](X), dtype=np.float32)
        pred = predictor_model.predict_step(torch.from_numpy(X), 0)
        pred = pred.detach().cpu().numpy()
        return self.transform[1](pred)

    def fit(self, X: NDArray[number]) -> "VaePredictorN":
        _, n_features = X.shape
        self.encoder_ref = self._build_encoder(n_features)
        self.decoder_ref = self._build_decoder(n_features)
        self._fit_vae(self.encoder_ref, self.decoder_ref, X)
        return self

    def predict(self, X: NDArray[number], indexer: NDArray[intp]) -> NDArray[number]:
        _, n_features = X.shape
        self.encoder_query = self._build_encoder(n_features)
        self.decoder_query = self._build_decoder(n_features)
        self._copy_decoder_for_query(indexer)
        self._freeze_decoder_query()
        self._fit_vae(self.encoder_query, self.decoder_query, X)
        res = self._predict_with_ref_decoder(X)
        return res

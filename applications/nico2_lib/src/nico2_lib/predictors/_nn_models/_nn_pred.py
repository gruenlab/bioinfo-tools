from dataclasses import dataclass, field, replace
from typing import Any, ClassVar, Dict, Optional, Tuple

import numpy as np
import torch
from numpy import intp, number
from numpy.typing import NDArray
from torch import nn

from nico2_lib.predictors._nn_models._base import BaseVAE
from nico2_lib.predictors._nn_models._nn import (
    LinearDecoder,
    LinearVariationalEncoder,
    NonLinearDecoder,
    NonLinearVariationalEncoder,
)


@dataclass(frozen=True, kw_only=True)
class VaePredictor:
    lr: float
    dataloader_kwargs: Dict[str, Any] = field(default_factory=dict)
    trainer_kwargs: Dict[str, Any] = field(default_factory=dict)
    model_ref: Optional[BaseVAE] = None

    _default_dataloader_kwargs: ClassVar[Dict[str, Any]] = {
        "batch_size": 64,
        "shuffle": True,
    }
    _default_trainer_kwargs: ClassVar[Dict[str, Any]] = {
        "max_epochs": 200,
        "enable_checkpointing": False,
        "logger": False,
        "gradient_clip_val": 1.0,
    }

    @property
    def _merged_dataloader_kwargs(self) -> Dict[str, Any]:
        return {**self._default_dataloader_kwargs, **self.dataloader_kwargs}

    @property
    def _merged_trainer_kwargs(self) -> Dict[str, Any]:
        return {**self._default_trainer_kwargs, **self.trainer_kwargs}

    def _build_model(self, n_features: int) -> BaseVAE:
        raise NotImplementedError("Subclasses must implement _build_model().")

    def _freeze_module(self, module: nn.Module) -> None:
        for param in module.parameters():
            param.requires_grad = False

    def _forward(
        self,
        X: NDArray[number],
        *,
        encoder: nn.Module,
        decoder: nn.Module,
        lr: float,
    ) -> Tuple[NDArray[number], NDArray[number]]:
        model = BaseVAE(encoder=encoder, decoder=decoder, lr=lr)
        embedding, prediction = model.predict_step(torch.from_numpy(X), 0)
        return embedding.detach().cpu().numpy(), prediction.detach().cpu().numpy()

    def fit(self, X: NDArray[number]) -> "VaePredictor":
        _, n_features = X.shape
        model_ref = self._build_model(n_features)
        model_ref.fit_vae(
            X,
            dataloader_kwargs=self._merged_dataloader_kwargs,
            trainer_kwargs=self._merged_trainer_kwargs,
        )
        return replace(self, model_ref=model_ref)

    def predict(
        self, X: NDArray[number], indexer: NDArray[intp]
    ) -> Tuple[NDArray[number], NDArray[number]]:
        if self.model_ref is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        _, n_features = X.shape
        model_query = self._build_model(n_features)
        decoder_query = self.model_ref.decoder.return_slice(indexer)
        model_query.decoder = decoder_query
        self._freeze_module(model_query.decoder)
        model_query.fit_vae(
            X,
            dataloader_kwargs=self._merged_dataloader_kwargs,
            trainer_kwargs=self._merged_trainer_kwargs,
        )
        X_prepared = np.asarray(X, dtype=np.float32)
        embedding, res = self._forward(
            X_prepared,
            encoder=model_query.encoder,
            decoder=self.model_ref.decoder,
            lr=model_query.lr,
        )
        return embedding, res


@dataclass(frozen=True, kw_only=True)
class FullyLinearVaePredictor(VaePredictor):
    latent_features: int

    def _build_model(self, n_features: int) -> BaseVAE:
        encoder = LinearVariationalEncoder(
            in_features=n_features,
            latent_features=self.latent_features,
        )
        decoder = LinearDecoder(
            latent_features=self.latent_features,
            out_features=n_features,
        )
        return BaseVAE(encoder=encoder, decoder=decoder, lr=self.lr)


@dataclass(frozen=True, kw_only=True)
class LinearlyEncodedVaePredictor(VaePredictor):
    latent_features: int
    hidden_features: int

    def _build_model(self, n_features: int) -> BaseVAE:
        encoder = LinearVariationalEncoder(
            in_features=n_features,
            latent_features=self.latent_features,
        )
        decoder = NonLinearDecoder(
            latent_features=self.latent_features,
            out_features=n_features,
            hidden_features=self.hidden_features,
        )
        return BaseVAE(encoder=encoder, decoder=decoder, lr=self.lr)


@dataclass(frozen=True, kw_only=True)
class LinearlyDecodedVaePredictor(VaePredictor):
    latent_features: int
    hidden_features: int

    def _build_model(self, n_features: int) -> BaseVAE:
        encoder = NonLinearVariationalEncoder(
            in_features=n_features,
            latent_features=self.latent_features,
            hidden_features=self.hidden_features,
        )
        decoder = LinearDecoder(
            latent_features=self.latent_features,
            out_features=n_features,
        )
        return BaseVAE(encoder=encoder, decoder=decoder, lr=self.lr)


@dataclass(frozen=True, kw_only=True)
class FullyNonlinearVaePredictor(VaePredictor):
    latent_features: int
    hidden_features: int

    def _build_model(self, n_features: int) -> BaseVAE:
        encoder = NonLinearVariationalEncoder(
            in_features=n_features,
            latent_features=self.latent_features,
            hidden_features=self.hidden_features,
        )
        decoder = NonLinearDecoder(
            latent_features=self.latent_features,
            out_features=n_features,
            hidden_features=self.hidden_features,
        )
        return BaseVAE(encoder=encoder, decoder=decoder, lr=self.lr)

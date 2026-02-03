from collections.abc import Callable

from nico2_lib.predictors._nn_models._base import BaseVAE
from nico2_lib.predictors._nn_models._nn import (
    LinearDecoder,
    LinearVariationalEncoder,
    NonLinearDecoder,
    NonLinearVariationalEncoder,
)


def fully_linear_vae(latent_features: int, lr: float) -> Callable[[int], BaseVAE]:
    """Return a factory for a fully linear VAE.

    Args:
        latent_features: Number of latent dimensions for the encoder/decoder.
        lr: Learning rate for the VAE optimizer.

    Returns:
        Callable that builds a `BaseVAE` given the input/output feature count.
    """

    def build(n_features: int) -> BaseVAE:
        encoder = LinearVariationalEncoder(
            in_features=n_features,
            latent_features=latent_features,
        )
        decoder = LinearDecoder(
            latent_features=latent_features,
            out_features=n_features,
        )
        return BaseVAE(encoder=encoder, decoder=decoder, lr=lr)

    return build


def linearily_encoded_vae(
    latent_features: int, hidden_features: int, lr: float
) -> Callable[[int], BaseVAE]:
    """Return a factory for a linear-encoder / nonlinear-decoder VAE."""

    def build(n_features: int) -> BaseVAE:
        encoder = LinearVariationalEncoder(
            in_features=n_features,
            latent_features=latent_features,
        )
        decoder = NonLinearDecoder(
            latent_features=latent_features,
            out_features=n_features,
            hidden_features=hidden_features,
        )
        return BaseVAE(encoder=encoder, decoder=decoder, lr=lr)

    return build


def linearily_decoded_vae(
    latent_features: int, hidden_features: int, lr: float
) -> Callable[[int], BaseVAE]:
    """Return a factory for a nonlinear-encoder / linear-decoder VAE."""

    def build(n_features: int) -> BaseVAE:
        encoder = NonLinearVariationalEncoder(
            in_features=n_features,
            latent_features=latent_features,
            hidden_features=hidden_features,
        )
        decoder = LinearDecoder(
            latent_features=latent_features,
            out_features=n_features,
        )
        return BaseVAE(encoder=encoder, decoder=decoder, lr=lr)

    return build


def fully_nonlinear_vae(
    latent_features: int, hidden_features: int, lr: float
) -> Callable[[int], BaseVAE]:
    """Return a factory for a fully nonlinear VAE."""

    def build(n_features: int) -> BaseVAE:
        encoder = NonLinearVariationalEncoder(
            in_features=n_features,
            latent_features=latent_features,
            hidden_features=hidden_features,
        )
        decoder = NonLinearDecoder(
            latent_features=latent_features,
            out_features=n_features,
            hidden_features=hidden_features,
        )
        return BaseVAE(encoder=encoder, decoder=decoder, lr=lr)

    return build
    


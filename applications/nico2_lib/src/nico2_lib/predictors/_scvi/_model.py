import torch
from scvi import REGISTRY_KEYS
from scvi.module.base import BaseModuleClass, LossOutput, auto_move_data
from torch.distributions import NegativeBinomial, Normal
from torch.distributions import kl_divergence as kl

from nico2_lib.predictors._scvi._nn import DecoderNet, EncoderNet


class recVAE(BaseModuleClass):
    """Skeleton Variational auto-encoder model.

    Here we implement a basic version of scVI's underlying VAE [Lopez18]_.
    This implementation is for instructional purposes only.

    Parameters
    ----------
    n_input
        Number of input genes.
    idx
        Subset of genes.
    n_latent
        Dimensionality of the latent space.
    """

    def __init__(
        self, n_input: int, idx: list, n_latent: int = 10, beta: float = 1, C: float = 0
    ):
        super().__init__()
        # in the init, we create the parameters of our elementary stochastic computation unit.

        # First, we setup the parameters of the generative model
        self.decoder = DecoderNet(n_latent, n_input, "softmax")
        self.log_theta = torch.nn.Parameter(torch.randn(n_input))

        # Second, we setup the parameters of the variational distribution
        self.mean_encoder = EncoderNet(len(idx), n_latent, "none")
        self.var_encoder = EncoderNet(len(idx), n_latent, "exp")

        self.idx = idx
        self.beta = beta
        self.C = C

    def _get_inference_input(
        self, tensors: dict[str, torch.Tensor], **kwargs
    ) -> dict[str, torch.Tensor]:
        """Parse the dictionary to get appropriate args"""
        # let us fetch the raw counts, and add them to the dictionary
        return {"x": tensors[REGISTRY_KEYS.X_KEY]}

    @auto_move_data
    def inference(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        High level inference method.

        Runs the inference (encoder) model.
        """

        # log the input to the variational distribution for numerical stability
        x_ = torch.log1p(x[:, self.idx])
        # get variational parameters via the encoder networks
        qz_m = self.mean_encoder(x_)
        qz_v = self.var_encoder(x_)
        # get one sample to feed to the generative model
        # under the hood here is the Reparametrization trick (Rsample)
        z = Normal(qz_m, torch.sqrt(qz_v)).rsample()

        library = torch.sum(x[:, self.idx], dim=1, keepdim=True)
        return {"qz_m": qz_m, "qz_v": qz_v, "z": z, "library": library}

    def _get_generative_input(
        self,
        tensors: dict[str, torch.Tensor],
        inference_outputs: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        return {
            "z": inference_outputs["z"],
            "library": torch.sum(tensors[REGISTRY_KEYS.X_KEY], dim=1, keepdim=True),
            # "library": inference_outputs["library"]
        }

    @auto_move_data
    def generative(
        self, z: torch.Tensor, library: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Runs the generative model."""
        # get the "normalized" mean of the negative binomial
        px_scale = self.decoder(z)
        # get the mean of the negative binomial
        px_rate = library * px_scale
        # get the dispersion parameter
        # theta = torch.exp(self.log_theta)
        theta = library

        return {
            "px_scale": px_scale,
            "theta": theta,
            "px_rate": px_rate,
        }

    def loss(
        self,
        tensors: dict[str, torch.Tensor],
        inference_outputs: dict[str, torch.Tensor],
        generative_outputs: dict[str, torch.Tensor],
    ) -> LossOutput:
        # here, we would like to form the ELBO. There are two terms:
        #   1. one that pertains to the likelihood of the data
        #   2. one that pertains to the variational distribution
        # so we extract all the required information
        x = tensors[REGISTRY_KEYS.X_KEY]
        px_rate = generative_outputs["px_rate"]
        theta = generative_outputs["theta"]
        qz_m = inference_outputs["qz_m"]
        qz_v = inference_outputs["qz_v"]

        # term 1
        # the pytorch NB distribution uses a different parameterization
        # so we must apply a quick transformation (included in scvi-tools, but here we use the
        # pytorch code)
        nb_logits = (px_rate + 1e-4).log() - (theta + 1e-4).log()
        log_lik = (
            NegativeBinomial(total_count=theta, logits=nb_logits)
            .log_prob(x)
            .sum(dim=-1)
        )

        # term 2
        prior_dist = Normal(torch.zeros_like(qz_m), torch.ones_like(qz_v))
        var_post_dist = Normal(qz_m, torch.sqrt(qz_v))
        kl_divergence = kl(var_post_dist, prior_dist).sum(dim=1)

        # beta-VAE-B
        # elbo = log_lik - self.beta * abs( kl_divergence - self.C)
        # beta-VAE
        # elbo = log_lik - self.beta * kl_divergence
        # VAE
        elbo = log_lik - kl_divergence
        loss = torch.mean(-elbo)
        return LossOutput(
            loss=loss,
            reconstruction_loss=-log_lik,
            kl_local=kl_divergence,
            kl_global=0.0,
        )

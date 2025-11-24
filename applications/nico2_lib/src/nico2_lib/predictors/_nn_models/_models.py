from ._nn import Decoder, LinearDecoder, VariationalEncoder, VariationalLinearEncoder
from ._base import BaseVAE


class LVAE(BaseVAE):
    """
    Linear encoder + linear decoder
    optuna best params per dataset:
        human_lung: {'latent_features': 16, 'lr': 0.001} # latent features 64 just as good
        h_embryo_11d: {'latent_features': 128, 'lr': 0.001} # latent_features 64 just as good

    """

    def __init__(self, input_features, output_features, latent_features=64, lr=1e-3):
        super().__init__(input_features, output_features, latent_features, lr)
        self.encoder = VariationalLinearEncoder(input_features, latent_features)
        self.decoder = LinearDecoder(latent_features, output_features)


class LEVAE(BaseVAE):
    """Linear encoder + nonlinear decoder"""

    def __init__(
        self,
        input_features,
        output_features,
        latent_features,
        hidden_features_out=None,
        lr=1e-4,
    ):
        super().__init__(input_features, output_features, latent_features, lr)
        self.encoder = VariationalLinearEncoder(input_features, latent_features)
        self.decoder = Decoder(latent_features, output_features, hidden_features_out)


class LDVAE(BaseVAE):
    """Nonlinear encoder + linear decoder"""

    def __init__(
        self,
        input_features,
        output_features,
        latent_features,
        hidden_features_in=None,
        lr=1e-4,
    ):
        super().__init__(input_features, output_features, latent_features, lr)
        self.encoder = VariationalEncoder(
            input_features, latent_features, hidden_features_in
        )
        self.decoder = LinearDecoder(latent_features, output_features)


class VAE(BaseVAE):
    """Nonlinear encoder + nonlinear decoder"""

    def __init__(
        self,
        input_features,
        output_features,
        latent_features,
        hidden_features_out=None,
        hidden_features_in=None,
        lr=1e-4,
    ):
        super().__init__(input_features, output_features, latent_features, lr)
        self.encoder = VariationalEncoder(
            input_features, latent_features, hidden_features_in
        )
        self.decoder = Decoder(latent_features, output_features, hidden_features_out)
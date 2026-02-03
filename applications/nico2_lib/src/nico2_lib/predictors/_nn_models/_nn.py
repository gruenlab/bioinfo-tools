from typing import Optional, Tuple, Union

import torch
from torch import Tensor, nn, relu


class LinearVariationalEncoder(nn.Module):
    """Linear encoder producing mu and logvar."""

    def __init__(self, in_features: int, latent_features: int) -> None:
        super().__init__()
        self.fc_mu = nn.Linear(in_features, latent_features, bias=False)
        self.fc_logvar = nn.Linear(in_features, latent_features, bias=True)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = mu + eps * std
        else:
            z = mu
        return z, mu, logvar


class NonLinearVariationalEncoder(nn.Module):
    """Nonlinear encoder producing mu and logvar."""

    def __init__(
        self,
        in_features: int,
        latent_features: int,
        hidden_features: int,
    ) -> None:
        super().__init__()
        self.hidden = nn.Linear(in_features, hidden_features)
        self.fc_mu = nn.Linear(hidden_features, latent_features)
        self.fc_logvar = nn.Linear(hidden_features, latent_features)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        h = relu(self.hidden(x))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = mu + eps * std
        else:
            z = mu
        return z, mu, logvar


class LinearDecoder(nn.Module):
    """Linear decoder — interpretable weights."""

    def __init__(self, latent_features: int, out_features: int) -> None:
        super().__init__()
        self.out_features = out_features
        self.latent_features = latent_features
        self.mu_out = nn.Linear(latent_features, out_features, bias=False)

    def forward(self, z: Tensor) -> Tensor:
        return self.mu_out(z)

    def return_slice(self, indexer) -> "LinearDecoder":
        decoder = LinearDecoder(
            latent_features=self.latent_features,
            out_features=len(indexer),
        )
        with torch.no_grad():
            decoder.mu_out.weight.copy_(self.mu_out.weight[torch.from_numpy(indexer)])
        return decoder


class NonLinearDecoder(nn.Module):
    """Nonlinear decoder with one hidden layer."""

    def __init__(
        self, latent_features: int, out_features: int, hidden_features: int
    ) -> None:
        super().__init__()
        self.latent_features = latent_features
        self.hidden_features = hidden_features
        self.out_features = out_features
        self.hidden = nn.Linear(latent_features, hidden_features)
        self.mu_out = nn.Linear(hidden_features, out_features)

    def forward(self, z: Tensor) -> Tensor:
        h = relu(self.hidden(z))
        return self.mu_out(h)

    def return_slice(self, indexer) -> "NonLinearDecoder":
        """returns decoder to return the correct features as per indexer"""
        decoder = NonLinearDecoder(
            latent_features=self.latent_features,
            out_features=len(indexer),
            hidden_features=self.hidden_features,
        )
        with torch.no_grad():
            decoder.hidden.weight.copy_(self.hidden.weight)
            decoder.hidden.bias.copy_(self.hidden.bias)
            decoder.mu_out.weight.copy_(self.mu_out.weight[torch.from_numpy(indexer)])
            decoder.mu_out.bias.copy_(self.mu_out.bias[torch.from_numpy(indexer)])
        return decoder


Encoder = Union[LinearVariationalEncoder, NonLinearVariationalEncoder]
Decoder = Union[LinearDecoder, NonLinearDecoder]

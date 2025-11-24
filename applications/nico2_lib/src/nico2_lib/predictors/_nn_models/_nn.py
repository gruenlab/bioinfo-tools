import torch
from torch import nn, relu


class VariationalLinearEncoder(nn.Module):
    """Linear encoder producing mu and logvar."""

    def __init__(self, in_features, latent_features):
        super().__init__()
        self.fc_mu = nn.Linear(in_features, latent_features, bias=False)
        self.fc_logvar = nn.Linear(in_features, latent_features, bias=True)

    def forward(self, x):
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = mu + eps * std
        else:
            z = mu
        return z, mu, logvar


class VariationalEncoder(nn.Module):
    """Nonlinear encoder producing mu and logvar."""

    def __init__(self, in_features, latent_features, hidden_features=None):
        super().__init__()
        hidden_features = hidden_features or (in_features + latent_features) // 2
        self.hidden = nn.Linear(in_features, hidden_features)
        self.fc_mu = nn.Linear(hidden_features, latent_features)
        self.fc_logvar = nn.Linear(hidden_features, latent_features)

    def forward(self, x):
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

    def __init__(self, latent_features, out_features):
        super().__init__()
        self.mu_out = nn.Linear(latent_features, out_features, bias=False)

    def forward(self, z):
        return self.mu_out(z)


class Decoder(nn.Module):
    """Nonlinear decoder with one hidden layer."""

    def __init__(self, latent_features, out_features, hidden_features=None):
        super().__init__()
        hidden_features = hidden_features or (latent_features + out_features) // 2
        self.hidden = nn.Linear(latent_features, hidden_features)
        self.mu_out = nn.Linear(hidden_features, out_features)

    def forward(self, z):
        h = relu(self.hidden(z))
        return self.mu_out(h)





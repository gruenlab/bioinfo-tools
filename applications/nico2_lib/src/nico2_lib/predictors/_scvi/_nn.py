from typing import Literal

import torch


class EncoderNet(torch.nn.Module):
    def __init__(
        self,
        n_input: int,
        n_output: int,
        link_var: Literal["exp", "none", "softmax"],
    ):
        """Encodes data of ``n_input`` dimensions into a space of ``n_output`` dimensions.

        Uses a one layer fully-connected neural network with 128 hidden nodes.

        Parameters
        ----------
        n_input
            The dimensionality of the input.
        n_output
            The dimensionality of the output.
        link_var
            The final non-linearity.
        """
        super().__init__()
        self.neural_net = torch.nn.Sequential(
            torch.nn.Linear(n_input, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, n_output),
        )
        self.transformation = None
        if link_var == "softmax":
            self.transformation = torch.nn.Softmax(dim=-1)
        elif link_var == "exp":
            self.transformation = torch.exp

    def forward(self, x: torch.Tensor):
        output = self.neural_net(x)
        if self.transformation:
            output = self.transformation(output)
        return output


# %%
class DecoderNet(torch.nn.Module):
    def __init__(
        self,
        n_input: int,
        n_output: int,
        link_var: Literal["exp", "none", "softmax"],
    ):
        """Encodes data of ``n_input`` dimensions into a space of ``n_output`` dimensions.

        Uses a one layer fully-connected neural network with 128 hidden nodes.

        Parameters
        ----------
        n_input
            The dimensionality of the input.
        n_output
            The dimensionality of the output.
        link_var
            The final non-linearity.
        """
        super().__init__()
        self.neural_net = torch.nn.Sequential(
            torch.nn.Linear(n_input, n_output)
        )
        self.transformation = None
        if link_var == "softmax":
            self.transformation = torch.nn.Softmax(dim=-1)
        elif link_var == "exp":
            self.transformation = torch.exp

    def forward(self, x: torch.Tensor):
        output = self.neural_net(x)
        if self.transformation:
            output = self.transformation(output)
        return output

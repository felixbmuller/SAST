import math

import torch
import torch.nn as nn
from torch import Tensor


class PositionalEncoding(nn.Module):
    """Positional Encoding from the Attention is All You Need paper.

    Based on https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    """

    pe: Tensor

    def __init__(self, d_model: int, max_len: int):
        """Init positional encoding

        Parameters
        ----------
        d_model : int
            last dimension of the tensors being processed
        max_len : int
            maximum sequence length that needs to be handled
        """
        super().__init__()

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )

        pe = torch.zeros(1, max_len, d_model)  # Tensor(t, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Apply positional encoding to a tensor.

        Parameters
        ----------
        x: Tensor(b, t, d_model)
            Tensor for which we calculate and apply the positional encoding.

        Returns
        -------
        Tensor(b, t, d_model)
            Sum of x and positional encoding of x
        """
        return x + self.pe[:, : x.shape[1]]

import math

import torch
import torch.nn as nn
from torch import Tensor

from transformer.utils import get_device


class PositionalEncoding(nn.Module):
    #
    #
    #  -------- __init__ -----------
    #
    def __init__(self,
                 emb_size: int,
                 dropout: float,
                 maxlen: int = 5000):
        super(PositionalEncoding, self).__init__()

        # den = torch.exp(- torch.arange(0, emb_size, 2) * math.log(10000) / emb_size)
        # pos = torch.arange(0, maxlen).reshape(maxlen, 1)

        # pos_embedding[:, 0::2] = torch.sin(pos * den)
        # pos_embedding[:, 1::2] = torch.cos(pos * den)
        self.pos_embedding = torch.ones((maxlen, emb_size)).unsqueeze(-2).to(get_device())

        self.dropout = nn.Dropout(dropout)
        # self.register_buffer('pos_embedding', pos_embedding)

    #
    #
    #  -------- forward -----------
    #
    def forward(self, token_embedding: Tensor):
        return self.dropout(token_embedding + self.pos_embedding[:token_embedding.size(0), :])

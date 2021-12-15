import math

import torch
import torch.nn as nn
from torch import Tensor

from transformer.utils import get_device


class TrainableEncoding(nn.Module):
    #
    #
    #  -------- __init__ -----------
    #
    def __init__(self,
                 emb_size: int,
                 dropout: float,
                 maxlen: int = 5000):
        super(TrainableEncoding, self).__init__()

        self.pos_embedding = torch.zeros((maxlen, emb_size)).unsqueeze(-2).to(get_device())
        self.dropout = nn.Dropout(dropout)

        self.init_weights()

    #
    #
    #  -------- forward -----------
    #
    def forward(self, token_embedding: Tensor):
        return self.dropout(token_embedding + self.pos_embedding[:token_embedding.size(0), :])

    #
    #
    #  -------- init_weights -----------
    #
    def init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)


class TrigonometricEncoding(nn.Module):
    #
    #
    #  -------- __init__ -----------
    #
    def __init__(self,
                 emb_size: int,
                 dropout: float,
                 maxlen: int = 5000):
        super(TrigonometricEncoding, self).__init__()

        den = torch.exp(- torch.arange(0, emb_size, 2) * math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)

        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    #
    #
    #  -------- forward -----------
    #
    def forward(self, token_embedding: Tensor):
        return self.dropout(token_embedding + self.pos_embedding[:token_embedding.size(0), :])
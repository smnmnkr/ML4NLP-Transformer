import math

import torch.nn as nn
from torch import Tensor


class TokenEmbedding(nn.Module):
    #
    #
    #  -------- __init__ -----------
    #
    def __init__(self, vocab_size: int, emb_size: int, dropout: float = 0.0):
        super(TokenEmbedding, self).__init__()

        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size

        self.dropout = nn.Dropout(dropout)

    #
    #
    #  -------- forward -----------
    #
    def forward(self, tokens: Tensor):
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)

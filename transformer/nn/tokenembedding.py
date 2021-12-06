import math

import torch.nn as nn
from torch import Tensor


class TokenEmbedding(nn.Module):
    #
    #
    #  -------- __init__ -----------
    #
    def __init__(self, vocab_size: int, emb_size):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size

    #
    #
    #  -------- forward -----------
    #
    def forward(self, tokens: Tensor):
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)

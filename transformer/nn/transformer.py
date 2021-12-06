import math

import torch.nn as nn
import torch.nn.functional as F

from .positionalencoding import PositionalEncoding


class Transformer(nn.Module):

    def __init__(
            self,
            num_tokens: int,
            num_outputs: int,
            dim_model: int = 64,
            num_heads: int = 2,
            num_encoder_layers: int = 3,
            num_decoder_layers: int = 3,
            dropout: float = 0.1,
    ):
        super().__init__()

        self.model_type = "Transformer"
        self.dim_model = dim_model

        self.positional_encoder = PositionalEncoding(
            dim_model=dim_model, dropout=dropout, max_len=5000
        )
        self.embedding = nn.Embedding(num_tokens, dim_model)
        self.transformer = nn.Transformer(
            d_model=dim_model,
            nhead=num_heads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dropout=dropout,
        )
        self.out = nn.Linear(dim_model, num_outputs)

    def forward(self, src, tgt, tgt_mask=None, src_pad_mask=None, tgt_pad_mask=None):
        # Src size must be (batch_size, src sequence length)
        # Tgt size must be (batch_size, tgt sequence length)

        # Embedding + positional encoding - Out size = (batch_size, sequence length, dim_model)
        src = self.embedding(src) * math.sqrt(self.dim_model)
        tgt = self.embedding(tgt) * math.sqrt(self.dim_model)
        src = self.positional_encoder(src)
        tgt = self.positional_encoder(tgt)

        # We could use the parameter batch_first=True, but our KDL version doesn't support it yet, so we permute
        # to obtain size (sequence length, batch_size, dim_model),
        src = src.permute(1, 0, 2)
        tgt = tgt.permute(1, 0, 2)

        # Transformer blocks - Out size = (sequence length, batch_size, num_tokens)
        transformer_out = self.transformer(
            src,
            tgt,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_pad_mask,
            tgt_key_padding_mask=tgt_pad_mask
        )

        return F.log_softmax(self.out(transformer_out), -1)

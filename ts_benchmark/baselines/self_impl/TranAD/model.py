import math

import torch
import torch.nn as nn
from torch.nn import TransformerDecoder, TransformerDecoderLayer, TransformerEncoder

from .dlutils import PositionalEncoding, TransformerEncoderLayer


class TranADModel(nn.Module):
    def __init__(self, n_features: int, win_size: int = 10, dropout: float = 0.1):
        super().__init__()
        self.name = "TranAD"
        self.n_feats = n_features
        self.n_window = win_size
        self.n = self.n_feats * self.n_window
        self.pos_encoder = PositionalEncoding(2 * n_features, dropout, self.n_window)
        encoder_layers = TransformerEncoderLayer(
            d_model=2 * n_features,
            nhead=n_features,
            dim_feedforward=16,
            dropout=dropout,
        )
        self.transformer_encoder = TransformerEncoder(encoder_layers, 1)
        decoder_layers1 = TransformerDecoderLayer(
            d_model=2 * n_features,
            nhead=n_features,
            dim_feedforward=16,
            dropout=dropout,
        )
        self.transformer_decoder1 = TransformerDecoder(decoder_layers1, 1)
        decoder_layers2 = TransformerDecoderLayer(
            d_model=2 * n_features,
            nhead=n_features,
            dim_feedforward=16,
            dropout=dropout,
        )
        self.transformer_decoder2 = TransformerDecoder(decoder_layers2, 1)
        self.fcn = nn.Sequential(nn.Linear(2 * n_features, n_features), nn.Sigmoid())

    def encode(self, src, c, tgt):
        src = torch.cat((src, c), dim=2)
        src = src * math.sqrt(self.n_feats)
        src = self.pos_encoder(src)
        memory = self.transformer_encoder(src)
        tgt = tgt.repeat(1, 1, 2)
        return tgt, memory

    def forward(self, x):
        src = x.transpose(0, 1)
        tgt = src[-1:, :, :]
        c = torch.zeros_like(src)
        x1 = self.fcn(self.transformer_decoder1(*self.encode(src, c, tgt)))
        c = (x1 - src[-1:, :, :]).pow(2).repeat(src.shape[0], 1, 1)
        x2 = self.fcn(self.transformer_decoder2(*self.encode(src, c, tgt)))
        out = src.clone()
        out[-1:, :, :] = x2
        phase1 = src.clone()
        phase1[-1:, :, :] = x1
        return phase1.transpose(0, 1), out.transpose(0, 1)

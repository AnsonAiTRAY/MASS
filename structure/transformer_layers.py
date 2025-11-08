import torch.nn as nn
import math
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class TransformerLayers(nn.Module):
    def __init__(self, hidden_dim=128, nlayers=4, mlp_ratio=4, num_heads=4, dropout=0.1):
        super().__init__()
        self.d_model = hidden_dim
        self.heads = num_heads

        encoder_layers = TransformerEncoderLayer(hidden_dim, num_heads, hidden_dim * mlp_ratio, dropout,
                                                 batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)

    def forward(self, src):
        B, L, D = src.shape
        src = src * math.sqrt(self.d_model)
        src = src.view(B, L, D)
        output = self.transformer_encoder(src)
        output = output.view(B, L, D)
        return output

import math
import torch
import torch.nn as nn


class Student(nn.Module):
    """Sorgusuz student: (1, T, 1024) -> logit (1, T).

    Ablation için ayarlanabilir: d (reduced dim), heads, layers, ffn, dropout, use_pos.
    Skor = sigmoid(logit). Logit KD (softmax-KL) için ayrıca kullanılır.
    """

    def __init__(self, d_in=1024, d=256, heads=4, layers=1, ffn=512,
                 dropout=0.5, attn_dropout=0.3, use_pos=True):
        super().__init__()
        assert d % heads == 0, 'd, heads ile bölünebilmeli'
        self.d = d
        self.use_pos = use_pos
        self.proj = nn.Sequential(nn.Linear(d_in, d), nn.LayerNorm(d), nn.Dropout(dropout))
        enc = nn.TransformerEncoderLayer(d, heads, dim_feedforward=ffn,
                                         dropout=attn_dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc, layers)
        self.head = nn.Sequential(nn.Linear(d, 128), nn.ReLU(), nn.Dropout(dropout),
                                  nn.Linear(128, 1))

    def pos_enc(self, T, device):
        pos = torch.arange(T, device=device).unsqueeze(1)
        div = torch.exp(torch.arange(0, self.d, 2, device=device) * (-math.log(10000.0) / self.d))
        pe = torch.zeros(T, self.d, device=device)
        pe[:, 0::2], pe[:, 1::2] = torch.sin(pos * div), torch.cos(pos * div)
        return pe

    def forward(self, x):                 # x: (1, T, 1024)
        h = self.proj(x)
        if self.use_pos:
            h = h + self.pos_enc(x.size(1), x.device)
        return self.head(self.encoder(h)).squeeze(-1)   # logit (1, T)

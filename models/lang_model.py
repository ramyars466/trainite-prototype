import torch
import torch.nn as nn
from trainite.models.registry import register_model

@register_model("lang_model")
class Model(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, num_heads=4, num_layers=2, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        # Add your model architecture here
        self.fc = nn.Linear(embed_dim, vocab_size)
        self.fc_out = self.fc # for vocab size discovery

    def forward(self, src, tgt):
        # Implement forward pass
        return self.fc(self.embedding(tgt))

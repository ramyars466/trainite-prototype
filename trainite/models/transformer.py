import math
import torch
import torch.nn as nn
from trainite.models.registry import register_model


class PositionalEncoding(nn.Module):

    def __init__(self, embed_dim, max_len=512):
        super().__init__()

        pe = torch.zeros(max_len, embed_dim)

        position = torch.arange(0, max_len).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, embed_dim, 2) *
            (-math.log(10000.0) / embed_dim)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):

        return x + self.pe[:, :x.size(1)]


@register_model("transformer")   # ← Auto-registers in MODEL_REGISTRY
class DecoderOnlyTransformer(nn.Module):

    def __init__(self, vocab_size, embed_dim, num_heads, num_layers, dropout):

        super().__init__()
 # Step 1: Embedding layer — converts token IDs to vectors
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        # Step 2: Add positional encoding
        self.pos_encoding = PositionalEncoding(embed_dim)
      # Step 3: Transformer decoder layers
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dropout=dropout,
            batch_first=True
        )

        self.decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_layers
        )
 # Step 4: Final linear layer — converts back to vocabulary probabilities
        self.fc_out = nn.Linear(embed_dim, vocab_size)

        self.embed_dim = embed_dim

    def forward(self, src, tgt):
# Convert source tokens to embeddings + add position info
        src_emb = self.pos_encoding(
            self.embedding(src) * math.sqrt(self.embed_dim)
        )

        tgt_emb = self.pos_encoding(
            self.embedding(tgt) * math.sqrt(self.embed_dim)
        )
  # Causal mask — prevents the model from "cheating" by looking ahead
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(
            tgt.size(1)
        ).to(src.device)
  # Run through transformer decoder
        output = self.decoder(
            tgt_emb,
            src_emb,
            tgt_mask=tgt_mask
        )

        # Convert to vocabulary probabilities
        return self.fc_out(output)
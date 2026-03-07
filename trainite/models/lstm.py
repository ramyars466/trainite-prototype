import torch
import torch.nn as nn

from trainite.models.registry import register_model


@register_model("lstm")
class LSTMModel(nn.Module):

    def __init__(self, vocab_size, d_model=128, num_layers=2):

        super().__init__()

        self.embedding = nn.Embedding(vocab_size, d_model)

        self.lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=num_layers,
            batch_first=True
        )

        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, src, tgt):

        x = self.embedding(tgt)

        output, _ = self.lstm(x)

        logits = self.fc_out(output)

        return logits
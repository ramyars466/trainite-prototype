import os

CONFIG_TEMPLATE = """model:
  name: "lang_model"
  params:
    embed_dim: 128
    num_heads: 4
    num_layers: 2
    dropout: 0.1

dataset:
  name: "{dataset_name}"
  params:
    vocab: "abcdefghijklmnopqrstuvwxyz"
    seq_length: 16
    dataset_size: 10000

training:
  max_epochs: 50
  batch_size: 32
  lr: 0.001
  output_dir: "output"
"""

MODEL_TEMPLATE = """import torch
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
"""

DATASET_TEMPLATE = """import torch
from torch.utils.data import Dataset
from trainite.datasets.registry import register_dataset

@register_dataset("{dataset_name}")
class Dataset(Dataset):
    def __init__(self, vocab, seq_length, dataset_size):
        self.vocab = vocab
        self.seq_length = seq_length
        self.dataset_size = dataset_size
        
    def __len__(self):
        return self.dataset_size
        
    def __getitem__(self, idx):
        # Return src, tgt tensors
        # src: input sequence, tgt: target sequence (with <SOS>)
        # We generate random integers between 1 and vocab_size to avoid 0 (padding index)
        # so that CrossEntropyLoss does not return NaN.
        vocab_len = len(self.vocab) + 2
        src = torch.randint(1, vocab_len, (self.seq_length,), dtype=torch.long)
        tgt = torch.randint(1, vocab_len, (self.seq_length + 1,), dtype=torch.long)
        return src, tgt
"""

MAIN_TEMPLATE = """import typer
from trainite.train import main as train_main

app = typer.Typer()

@app.command()
def run(config: str = "configs/config.yaml"):
    \"\"\"
    Run training for the local project
    \"\"\"
    train_main(config)

if __name__ == "__main__":
    app()
"""

def scaffold_project(dataset_name="my_dataset"):
    os.makedirs("configs", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    os.makedirs("datasets", exist_ok=True)

    with open("configs/config.yaml", "w") as f:
        f.write(CONFIG_TEMPLATE.format(dataset_name=dataset_name))

    with open("models/lang_model.py", "w") as f:
        f.write(MODEL_TEMPLATE)

    with open(f"datasets/str_rev_dataset.py", "w") as f:
        f.write(DATASET_TEMPLATE.format(dataset_name=dataset_name))

    with open("main.py", "w") as f:
        f.write(MAIN_TEMPLATE)

import torch
from torch.utils.data import Dataset
from trainite.datasets.registry import register_dataset

@register_dataset("string-reverse")
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
        src = torch.zeros(self.seq_length, dtype=torch.long)
        tgt = torch.zeros(self.seq_length + 1, dtype=torch.long)
        return src, tgt

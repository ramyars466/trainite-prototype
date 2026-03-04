import random
import torch
from torch.utils.data import Dataset


class StringReversalDataset(Dataset):

    def __init__(self, vocab, seq_len, size, seed=42):

        self.vocab = vocab
        self.seq_len = seq_len

        self.char2idx = {"<PAD>": 0, "<SOS>": 1}
        for i, c in enumerate(vocab):
            self.char2idx[c] = i + 2

        self.idx2char = {v: k for k, v in self.char2idx.items()}

        random.seed(seed)

        self.data = []
        for _ in range(size):
            seq = random.choices(vocab, k=seq_len)
            self.data.append(seq)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        seq = self.data[idx]
        reversed_seq = seq[::-1]

        src = torch.tensor([self.char2idx[c] for c in seq])

        tgt = torch.tensor(
            [self.char2idx["<SOS>"]] +
            [self.char2idx[c] for c in reversed_seq]
        )

        return src, tgt

    def decode(self, indices):

        return "".join(
            [self.idx2char.get(i.item(), "?")
             for i in indices if i.item() > 1]
        )
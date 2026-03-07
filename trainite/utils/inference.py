import torch
import yaml
import os
import glob

from trainite.models.registry import get_model


def get_latest_checkpoint(output_dir="output"):
    """
    Automatically find the latest checkpoint in the output directory
    """

    checkpoint_files = glob.glob(os.path.join(output_dir, "*.pt"))

    if len(checkpoint_files) == 0:
        raise FileNotFoundError("No checkpoint found in output directory")

    # sort by modification time
    checkpoint_files.sort(key=os.path.getmtime)

    latest_checkpoint = checkpoint_files[-1]

    print(f"Loading checkpoint: {latest_checkpoint}")

    return latest_checkpoint


def load_model(checkpoint_path=None, config_path="trainite/configs/config.yaml"):

    # If checkpoint not provided → automatically detect
    if checkpoint_path is None:
        checkpoint_path = get_latest_checkpoint()

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    vocab = list(config["dataset"]["params"]["vocab"])
    seq_len = config["dataset"]["params"]["seq_length"]

    vocab_size = len(vocab) + 2

    ModelClass = get_model(config["model"]["name"])

    model = ModelClass(
        vocab_size=vocab_size,
        **config["model"]["params"]
    )

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint)

    model.eval()

    return model, vocab, seq_len


def predict_reverse(model, vocab, seq_len, text):

    stoi = {ch: i + 2 for i, ch in enumerate(vocab)}
    stoi["<pad>"] = 0
    stoi["<bos>"] = 1

    itos = {i: ch for ch, i in stoi.items()}

    # encode source
    src_tokens = [stoi[c] for c in text if c in stoi]
    src_tokens = src_tokens[:seq_len]

    while len(src_tokens) < seq_len:
        src_tokens.append(0)

    src = torch.tensor(src_tokens).unsqueeze(0)

    # start decoder with BOS
    generated = [stoi["<bos>"]]

    for _ in range(seq_len):

        tgt = torch.tensor(generated).unsqueeze(0)

        with torch.no_grad():
            output = model(src, tgt)

        next_token = torch.argmax(output[0, -1]).item()

        generated.append(next_token)

    # convert tokens to text
    result = ""

    for token in generated[1:]:
        if token in itos:
            result += itos[token]

    # remove extra predicted tokens (keep only reversed length)
    result = result[-len(text):]

    return result
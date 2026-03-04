import yaml
import torch

from torch.utils.data import DataLoader

from trainite.datasets.string_reverse import StringReversalDataset
from trainite.models.transformer import DecoderOnlyTransformer
from trainite.trainers.ignite_trainer import create_trainer


def load_config(path):

    with open(path, "r") as f:
        config = yaml.safe_load(f)

    return config


def main():

    config = load_config("trainite/configs/config.yaml")

    vocab = list(config["vocab"])

    dataset = StringReversalDataset(
        vocab,
        config["seq_len"],
        config["dataset_size"]
    )

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset,
        [train_size, val_size]
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"]
    )

    vocab_size = len(vocab) + 2

    model = DecoderOnlyTransformer(
        vocab_size=vocab_size,
        embed_dim=config["embed_dim"],
        num_heads=config["num_heads"],
        num_layers=config["num_layers"],
        dropout=config["dropout"]
    )

    trainer = create_trainer(
        model,
        train_loader,
        val_loader,
        config
    )

    trainer.run(
        train_loader,
        max_epochs=config["max_epochs"]
    )


if __name__ == "__main__":
    main()
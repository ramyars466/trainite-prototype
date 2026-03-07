import yaml
import torch
import trainite.models
import trainite.models.transformer
from torch.utils.data import DataLoader

from trainite.datasets.registry import get_dataset
from trainite.models.registry import get_model
from trainite.trainers.ignite_trainer import create_trainer
from trainite.utils.experiment import create_experiment, log_metrics
from trainite.utils.experiment import create_experiment


def load_config(path):

    with open(path, "r") as f:
        config = yaml.safe_load(f)

    return config


def main(config_path):

    config = load_config(config_path)

    run_dir = create_experiment(config)
    print(f"Experiment directory created at: {run_dir}")

    vocab = list(config["dataset"]["params"]["vocab"])

    DatasetClass = get_dataset(config["dataset"]["name"])

    dataset = DatasetClass(
        config["dataset"]["params"]["vocab"],
        config["dataset"]["params"]["seq_length"],
        config["dataset"]["params"]["dataset_size"]
    )

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset,
        [train_size, val_size]
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config["training"]["batch_size"]
    )

    vocab_size = len(vocab) + 2

    ModelClass = get_model(config["model"]["name"])

    model = ModelClass(
        vocab_size=vocab_size,
        **config["model"]["params"]
    )

    trainer = create_trainer(
        model,
        train_loader,
        val_loader,
        config,
        run_dir
    )

    trainer.run(
        train_loader,
        max_epochs=config["training"]["max_epochs"]
    )


if __name__ == "__main__":
    main()
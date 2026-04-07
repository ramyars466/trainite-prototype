import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import yaml
import torch
import trainite.models
import trainite.models.transformer
from torch.utils.data import DataLoader

from trainite.datasets.registry import get_dataset
from trainite.models.registry import get_model
from trainite.trainers.ignite_trainer import create_trainer
from trainite.datasets.registry import load_plugin_datasets
from trainite.models.registry import load_plugin_models
from trainite.utils.experiment import create_experiment, save_config, log_metrics


def load_config(path):

    with open(path, "r") as f:
        config = yaml.safe_load(f)

    return config


def main(config_path):

    config = load_config(config_path)

    run_dir = create_experiment(config)
    save_config(run_dir, config)
    
    load_plugin_datasets()
    load_plugin_models()

    
    

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
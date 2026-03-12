import os
import json
from datetime import datetime


def create_experiment(config):
    """
    Create a new experiment run directory.
    """

    base_dir = "experiments"
    os.makedirs(base_dir, exist_ok=True)

    run_name = datetime.now().strftime("run_%Y%m%d_%H%M%S")

    run_dir = os.path.join(base_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)

    print(f"Experiment directory created at: {run_dir}")

    return run_dir


def save_config(run_dir, config):
    """
    Save experiment configuration.
    """

    config_path = os.path.join(run_dir, "config.json")

    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)


def log_metrics(run_dir, metrics):
    """
    Save training metrics.
    """

    metrics_path = os.path.join(run_dir, "metrics.json")

    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)


def save_model(run_dir, model_state):
    """
    Save trained model weights.
    """

    import torch

    model_path = os.path.join(run_dir, "model.pt")

    torch.save(model_state, model_path)
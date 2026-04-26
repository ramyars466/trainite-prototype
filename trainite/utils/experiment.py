import os
import json
import shutil
import glob
from datetime import datetime


def create_experiment(config):
    """
    Create a new experiment run directory.
    """
 # Creates: experiments/run_20260403_143000/
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
     # Saves: experiments/run_20260403_143000/config.json

    config_path = os.path.join(run_dir, "config.json")

    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)


def log_metrics(run_dir, metrics):
    """
    Save training metrics.
    """
    # Saves: experiments/run_20260403_143000/metrics.json
    # {"train_loss": [0.9, 0.7, 0.5, ...], "val_loss": [0.8, 0.6, ...]}

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


def save_code_snapshot(run_dir, config_path):
    """
    Save a snapshot of the code used for this experiment.
    """
    code_dir = os.path.join(run_dir, "code")
    os.makedirs(code_dir, exist_ok=True)
    
    if os.path.exists(config_path):
        shutil.copy(config_path, code_dir)
        
    if os.path.exists("main.py"):
        shutil.copy("main.py", code_dir)
        
    for py_file in glob.glob("models/*.py") + glob.glob("datasets/*.py"):
        dest_dir = os.path.join(code_dir, os.path.dirname(py_file))
        os.makedirs(dest_dir, exist_ok=True)
        shutil.copy(py_file, os.path.join(dest_dir, os.path.basename(py_file)))
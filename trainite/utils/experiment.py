import os
import json
import datetime
import shutil


def create_experiment(config):

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    run_dir = os.path.join("experiments", f"run_{timestamp}")

    os.makedirs(run_dir, exist_ok=True)

    # save config
    config_path = os.path.join(run_dir, "config.yaml")

    try:
        import yaml
        with open(config_path, "w") as f:
            yaml.dump(config, f)
    except Exception:
        pass

    print(f"Experiment directory created at: {run_dir}")

    return run_dir


def log_metrics(run_dir, metrics):

    metrics_file = os.path.join(run_dir, "metrics.json")

    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=4)
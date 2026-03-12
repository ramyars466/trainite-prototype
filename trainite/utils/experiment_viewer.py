import os
import json


def list_experiments():

    base_dir = "experiments"

    if not os.path.exists(base_dir):
        print("No experiments found.")
        return

    runs = sorted(os.listdir(base_dir))

    if len(runs) == 0:
        print("No experiment runs available.")
        return

    print("\nAvailable Experiments\n")

    for run in runs:
        print(run)


def show_experiment(run_name):

    run_dir = os.path.join("experiments", run_name)

    if not os.path.exists(run_dir):
        print("Experiment not found.")
        return

    print(f"\nExperiment: {run_name}\n")

    config_path = os.path.join(run_dir, "config.json")
    metrics_path = os.path.join(run_dir, "metrics.json")

    if os.path.exists(config_path):

        print("Config\n")

        with open(config_path) as f:
            config = json.load(f)

        print(json.dumps(config, indent=4))

    if os.path.exists(metrics_path):

        print("\nMetrics\n")

        with open(metrics_path) as f:
            metrics = json.load(f)

        print(json.dumps(metrics, indent=4))

def compare_experiments():
    
    base_dir = "experiments"

    if not os.path.exists(base_dir):
        print("No experiments found.")
        return

    runs = sorted(os.listdir(base_dir))

    print("\nExperiment Comparison\n")
    print(f"{'Run':30} {'Best Val Loss'}")
    print("-" * 45)

    for run in runs:

        metrics_path = os.path.join(base_dir, run, "metrics.json")

        if not os.path.exists(metrics_path):
            continue

        with open(metrics_path) as f:
            metrics = json.load(f)

        val_losses = metrics.get("val_loss", [])

        if len(val_losses) == 0:
            continue

        best_loss = min(val_losses)

        print(f"{run:30} {best_loss:.6f}")
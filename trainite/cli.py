import typer
import json
import os

from trainite.train import main as train_main
from trainite.utils.inference import load_model, predict_reverse
from trainite.utils.experiment_viewer import (
    list_experiments,
    show_experiment,
    compare_experiments
)

app = typer.Typer()


@app.command()
def train(config: str = "trainite/configs/config.yaml"):
    """
    Train the model using config.yaml
    """
    train_main(config)


@app.command()
def generate(text: str):
    """
    Generate text using trained model
    Example:
    trainite generate hello
    """

    model, vocab, seq_len = load_model()

    generated = predict_reverse(model, vocab, seq_len, text)

    print("\nGenerated Text:")
    print(generated)


@app.command()
def register_dataset(path: str):
    """
    Register an external dataset plugin.
    Example:
    trainite register-dataset my_dataset.py
    """

    plugins_file = "trainite/plugins/datasets.json"

    # create plugin directory if not exists
    os.makedirs("trainite/plugins", exist_ok=True)

    # create json if not exists
    if not os.path.exists(plugins_file):
        with open(plugins_file, "w") as f:
            json.dump({}, f)

    # load existing datasets
    with open(plugins_file, "r") as f:
        datasets = json.load(f)

    name = os.path.basename(path).replace(".py", "")

    datasets[name] = path

    # save dataset registry
    with open(plugins_file, "w") as f:
        json.dump(datasets, f, indent=4)

    print(f"Dataset '{name}' registered successfully.")


@app.command()
def register_model(path: str):
    """
    Register an external model plugin.
    Example:
    trainite register-model my_model.py
    """

    plugins_file = "trainite/plugins/models.json"

    os.makedirs("trainite/plugins", exist_ok=True)

    if not os.path.exists(plugins_file):
        with open(plugins_file, "w") as f:
            json.dump({}, f)

    with open(plugins_file, "r") as f:
        models = json.load(f)

    name = os.path.basename(path).replace(".py", "")

    models[name] = path

    with open(plugins_file, "w") as f:
        json.dump(models, f, indent=4)

    print(f"Model '{name}' registered successfully.")


@app.command()
def experiments():
    """
    List all experiment runs
    """

    list_experiments()


@app.command()
def experiment(name: str):
    """
    Show details of an experiment run
    """

    show_experiment(name)

@app.command()
def compare():
    """
    Compare experiment runs based on validation loss
    """

    compare_experiments()


if __name__ == "__main__":
    app()
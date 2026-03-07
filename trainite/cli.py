import typer

from trainite.train import main as train_main
from trainite.utils.inference import load_model, predict_reverse

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
    python -m trainite.cli generate hello
    """

    

    model, vocab, seq_len = load_model()

    generated = predict_reverse(model, vocab, seq_len, text)

    print("\nGenerated Text:")
    print(generated)


if __name__ == "__main__":
    app()
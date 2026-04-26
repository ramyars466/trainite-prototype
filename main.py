import typer
from trainite.train import main as train_main

app = typer.Typer()

@app.command()
def run(config: str = "configs/config.yaml"):
    """
    Run training for the local project
    """
    train_main(config)

if __name__ == "__main__":
    app()

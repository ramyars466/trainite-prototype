import os
import torch
import torch.nn as nn

from ignite.engine import Engine, Events
from ignite.handlers import ModelCheckpoint, EarlyStopping, global_step_from_engine
from torch.utils.tensorboard import SummaryWriter

from trainite.utils.experiment import log_metrics

#setup: this is where we define the training loop
def create_trainer(model, train_loader, val_loader, config, run_dir):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)   # Move model to GPU/CPU

    optimizer = torch.optim.Adam( # Adam optimizer
        model.parameters(),
        lr=config["training"]["lr"],
    )

    criterion = nn.CrossEntropyLoss(ignore_index=0)

    writer = SummaryWriter("runs/trainite_experiment")

    os.makedirs(config["training"]["output_dir"], exist_ok=True)

    vocab_size = model.fc_out.out_features

    # store metrics
    metrics_log = {
        "train_loss": [],
        "val_loss": []
    }

    # -----------------------------
    # TRAIN STEP
    # -----------------------------

    def train_step(engine, batch):

        model.train() # Set model to training mode

        src, tgt = [x.to(device) for x in batch] # Get input & target

        tgt_input = tgt[:, :-1] # Target WITHOUT last token (what model sees)
        tgt_label = tgt[:, 1:]# Target WITHOUT first token (what model predicts)

        optimizer.zero_grad() # Reset gradients

        output = model(src, tgt_input) # Forward pass

        loss = criterion(
            output.reshape(-1, vocab_size),
            tgt_label.reshape(-1)
        ) # Calculate loss

        loss.backward() # Backpropagation
        optimizer.step() # Update weights

        writer.add_scalar("train/loss", loss.item(), engine.state.iteration) # Log loss

        metrics_log["train_loss"].append(loss.item())

        return loss.item()

    # -----------------------------
    # EVAL STEP
    # -----------------------------

    def eval_step(engine, batch):

        model.eval()

        with torch.no_grad():

            src, tgt = [x.to(device) for x in batch]

            tgt_input = tgt[:, :-1]
            tgt_label = tgt[:, 1:]

            output = model(src, tgt_input)

            loss = criterion(
                output.reshape(-1, vocab_size),
                tgt_label.reshape(-1)
            )

        metrics_log["val_loss"].append(loss.item())

        return loss.item()

    trainer = Engine(train_step)
    evaluator = Engine(eval_step)

    # -----------------------------
    # TRAIN LOG
    # -----------------------------
# HANDLER 1: After every epoch, log loss and run validation
    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training(engine):

        print(f"Epoch {engine.state.epoch} | Train Loss: {engine.state.output:.4f}")

        evaluator.run(val_loader) # Trigger validation

    # -----------------------------
    # VALIDATION LOG
    # -----------------------------
# HANDLER 2: After validation completes, print val loss
    @evaluator.on(Events.COMPLETED)
    def log_validation(engine):

        val_loss = engine.state.output

        print(f"           | Val Loss: {val_loss:.4f}")

        writer.add_scalar("validation/loss", val_loss, trainer.state.epoch)

    # -----------------------------
    # EARLY STOPPING
    # -----------------------------
 # an early stopping handler monitors the validation loss and terminates training if it stops improving for 5 epochs
    def score_function(engine):
        val_loss = engine.state.output
        return -val_loss

    early_stopping = EarlyStopping(
        patience=5,
        score_function=score_function,
        trainer=trainer
    )

    evaluator.add_event_handler(Events.COMPLETED, early_stopping)

    # -----------------------------
    # MODEL CHECKPOINT
    # -----------------------------
# HANDLER 4: Save best model checkpoint
    checkpoint = ModelCheckpoint(
        config["training"]["output_dir"],
        filename_prefix="best_model",
        n_saved=1,
        score_function=score_function,
        score_name="val_loss",
        global_step_transform=global_step_from_engine(trainer),
        require_empty=False
    )

    evaluator.add_event_handler(
        Events.COMPLETED,
        checkpoint,
        {"model": model}
    )

    # -----------------------------
    # SAVE METRICS AFTER TRAINING
    # -----------------------------

    @trainer.on(Events.COMPLETED)
    def save_metrics(engine):
        log_metrics(run_dir, metrics_log)

    return trainer
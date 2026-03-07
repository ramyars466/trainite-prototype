import os
import torch
import torch.nn as nn

from ignite.engine import Engine, Events
from ignite.handlers import ModelCheckpoint, EarlyStopping, global_step_from_engine
from torch.utils.tensorboard import SummaryWriter

from trainite.utils.experiment import log_metrics


def create_trainer(model, train_loader, val_loader, config, run_dir):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    optimizer = torch.optim.Adam(
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

        model.train()

        src, tgt = [x.to(device) for x in batch]

        tgt_input = tgt[:, :-1]
        tgt_label = tgt[:, 1:]

        optimizer.zero_grad()

        output = model(src, tgt_input)

        loss = criterion(
            output.reshape(-1, vocab_size),
            tgt_label.reshape(-1)
        )

        loss.backward()
        optimizer.step()

        writer.add_scalar("train/loss", loss.item(), engine.state.iteration)

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

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training(engine):

        print(f"Epoch {engine.state.epoch} | Train Loss: {engine.state.output:.4f}")

        evaluator.run(val_loader)

    # -----------------------------
    # VALIDATION LOG
    # -----------------------------

    @evaluator.on(Events.COMPLETED)
    def log_validation(engine):

        val_loss = engine.state.output

        print(f"           | Val Loss: {val_loss:.4f}")

        writer.add_scalar("validation/loss", val_loss, trainer.state.epoch)

    # -----------------------------
    # EARLY STOPPING
    # -----------------------------

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
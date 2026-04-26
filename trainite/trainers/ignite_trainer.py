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
   
    vocab_size = model.fc_out.out_features

    metrics_log = {
        "train_loss": [],
        "val_loss": []
    }

    def train_step(engine, batch):
        model.train()
        src, tgt = [x.to(device) for x in batch]
        tgt_input = tgt[:, :-1]
        tgt_label = tgt[:, 1:]
        optimizer.zero_grad()
        output = model(src, tgt_input)
        loss = criterion(output.reshape(-1, vocab_size), tgt_label.reshape(-1))
        loss.backward()
        optimizer.step()
        writer.add_scalar("train/loss", loss.item(), engine.state.iteration)
        return loss.item()

    def eval_step(engine, batch):
        model.eval()
        with torch.no_grad():
            src, tgt = [x.to(device) for x in batch]
            tgt_input = tgt[:, :-1]
            tgt_label = tgt[:, 1:]
            output = model(src, tgt_input)
            loss = criterion(output.reshape(-1, vocab_size), tgt_label.reshape(-1))
        return loss.item()

    trainer = Engine(train_step)
    evaluator = Engine(eval_step)

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training(engine):
        avg_train_loss = engine.state.output
        print(f"Epoch {engine.state.epoch} | Train Loss: {avg_train_loss:.4f}")
        metrics_log["train_loss"].append(round(avg_train_loss, 4))
        evaluator.run(val_loader)

    @evaluator.on(Events.COMPLETED)
    def log_validation(engine):
        val_loss = engine.state.output
        print(f"           | Val Loss: {val_loss:.4f}")
        metrics_log["val_loss"].append(round(val_loss, 4))
        writer.add_scalar("validation/loss", val_loss, trainer.state.epoch)

    def score_function(engine):
        return -engine.state.output

    early_stopping = EarlyStopping(
        patience=5,
        score_function=score_function,
        trainer=trainer
    )
    evaluator.add_event_handler(Events.COMPLETED, early_stopping)

    @evaluator.on(Events.COMPLETED)
    def save_best_model(engine):
        val_loss = engine.state.output
        best_loss = getattr(engine.state, "best_loss", float("inf"))
        if val_loss < best_loss:
            engine.state.best_loss = val_loss
            torch.save(model.state_dict(), os.path.join(run_dir, "best.pt"))
            
    @trainer.on(Events.EPOCH_COMPLETED)
    def save_last_model(engine):
        torch.save(model.state_dict(), os.path.join(run_dir, "last.pt"))

    @trainer.on(Events.COMPLETED)
    def save_metrics(engine):
        metrics_log["best_val_loss"] = min(metrics_log["val_loss"]) if metrics_log["val_loss"] else None
        metrics_log["total_epochs"] = engine.state.epoch
        log_metrics(run_dir, metrics_log)

    return trainer

    #trainite --help
    #type trainite\configs\config.yaml
    #trainite train trainite/configs/config.yaml
    #trainite experiments
    #trainite compare
    #dir experiments\run_20260407_214518
    #type experiments\run_20260407_214518\metrics.json
    #type experiments\run_20260407_214518\config.json
    #trainite experiment run_20260407_214518
    #tensorboard --logdir runs
import os
import torch
import torch.nn as nn

from ignite.engine import Engine, Events
from ignite.handlers import ModelCheckpoint, global_step_from_engine


def create_trainer(
    model,
    train_loader,
    val_loader,
    config,
):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["lr"]
    )

    criterion = nn.CrossEntropyLoss(ignore_index=0)

    os.makedirs(config["output_dir"], exist_ok=True)

    vocab_size = model.fc_out.out_features


    # -----------------------
    # TRAIN STEP
    # -----------------------

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

        return loss.item()


    # -----------------------
    # EVAL STEP
    # -----------------------

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

        return loss.item()


    trainer = Engine(train_step)

    evaluator = Engine(eval_step)


    # -----------------------
    # LOGGING
    # -----------------------

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training(engine):

        print(f"Epoch {engine.state.epoch} | Train Loss: {engine.state.output:.4f}")

        evaluator.run(val_loader)


    @evaluator.on(Events.COMPLETED)
    def log_validation(engine):

        print(f"           | Val Loss: {engine.state.output:.4f}")


    # -----------------------
    # CHECKPOINT
    # -----------------------

    checkpoint = ModelCheckpoint(
        config["output_dir"],
        filename_prefix="best",
        n_saved=1,
        score_function=lambda e: -evaluator.state.output,
        score_name="val_loss",
        global_step_transform=global_step_from_engine(trainer),
        require_empty=False
    )

    evaluator.add_event_handler(
        Events.COMPLETED,
        checkpoint,
        {"model": model}
    )


    return trainer
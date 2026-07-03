from pathlib import Path
from statistics import mean
import sys
import torch
import torch.utils.data as data
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from sast.realism_classifier.config import get_cfg_defaults
from sast.realism_classifier.data import load_dataset
from sast.realism_classifier.model import RealismClassifier
from sast.utils import count_parameters


def train():
    cfg = get_cfg_defaults()
    cfg.merge_from_list(sys.argv[1:])

    data_path = Path(cfg.data_path)

    print("Loading training data")

    train_dl = data.DataLoader(
        load_dataset(f"{data_path}/realism_train.npz", cfg.device),
        batch_size=cfg.train_batch_size,
        shuffle=True,
    )

    print(f"Training samples: {len(train_dl.dataset)}")

    n_batches = len(train_dl.dataset) // cfg.train_batch_size

    print("Loading validation data")

    val_gt_dl = data.DataLoader(
        load_dataset(f"{data_path}/gt_test.npz", cfg.device),
        batch_size=cfg.val_batch_size,
        shuffle=False,
    )

    val_models_dl = data.DataLoader(
        load_dataset(f"{data_path}/synthetic_val.npz", cfg.device),
        batch_size=cfg.val_batch_size,
        shuffle=False,
    )

    print("Creating model")

    model = RealismClassifier(cfg).to(cfg.device)

    writer = SummaryWriter()

    save_dir = Path(writer.get_logdir())

    writer.add_text("parameter_count", f"{count_parameters(model)}")
    writer.add_text("pytorch_version", str(torch.__version__))

    print("Model parameter ", count_parameters(model))

    with open(save_dir / "config.yaml", "w") as fp:
        fp.write(cfg.dump())

    print("Setup optimizer")

    optim, _ = model.configure_optimizers()

    global_step = 0

    for epoch in range(cfg.n_epochs):
        total_loss = 0

        for idx, batch in tqdm(enumerate(train_dl), total=n_batches, desc="Train"):
            loss = model.training_step(batch, idx)

            optim.zero_grad()
            loss.backward()
            optim.step()

            writer.add_scalar("StepLoss/Train", loss.item(), global_step)
            writer.add_scalar("Epoch", epoch, global_step)

            total_loss += loss.item()
            global_step += 1

        with torch.no_grad():
            val_gt_loss = 0

            for idx, batch in tqdm(enumerate(val_gt_dl), total=len(val_gt_dl), desc="Val/GT"):
                X, y = batch

                loss = model.loss(model.forward(X), y)

                val_gt_loss += loss.item()

            val_models_loss = 0

            for idx, batch in tqdm(enumerate(val_models_dl), total=len(val_models_dl), desc="Val/Syn"):
                X, y = batch

                loss = model.loss(model.forward(X), y)

                val_models_loss += loss.item()

                # TODO add ROC-AUC support

        if True:
            torch.save(
                {
                    "cfg": cfg,
                    "model_state": model.state_dict(),
                    "optim_state": optim.state_dict(),
                    "epoch": epoch,
                },
                save_dir / f"save_{epoch=:03}",
            )

        writer.add_scalar("Loss/Train", total_loss, global_step)
        writer.add_scalar("LossReal/Val", val_gt_loss, global_step)
        writer.add_scalar("LossSynthetic/Val", val_models_loss, global_step)
        writer.add_scalar("Loss/Val", mean([val_models_loss, val_gt_loss]), global_step)

        print(
            f"{epoch=}, {total_loss=}, {val_gt_loss=}, {val_models_loss=}, "
            f"val_combined_loss={mean([val_models_loss, val_gt_loss])}"
        )

    writer.close()


if __name__ == "__main__":
    train()

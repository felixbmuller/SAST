import logging
from pprint import pformat

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

from sast.utils import (
    BASE_PATH,
    count_parameters,
    get_git_revision_hash,
    load_config,
    set_seed,
)
from sast.model.tcn import MultiTcnDiffusion
from sast.data.uniform_data_module import UniformDataModule


def main(cfg):

    set_seed(42)

    print("Load data")
    data_module = UniformDataModule(cfg)

    print("SAMPLE COUNT", data_module.num_samples())

    statistics = data_module.calculate_train_statistics()

    model = MultiTcnDiffusion(
        cfg,
        data_mean=statistics["mean"],
        data_std=statistics["std"],
    )

    additional_params = {
        "git_hash": get_git_revision_hash(),
        "num_samples": data_module.num_samples(),
        "model_params": count_parameters(model),
    }

    logger = TensorBoardLogger(
        str(BASE_PATH / "experiments" / cfg.experiment.study_name),
        cfg.experiment.run_name,
        log_graph=False,
    )

    logger.log_hyperparams(additional_params)
    logger.experiment.add_text("config", pformat(cfg), 0)
    logger.experiment.add_text("additional_params", pformat(additional_params), 0)

    checkpoint_callback = ModelCheckpoint(
        save_top_k=-1,
        every_n_epochs=cfg.experiment.chkpt_every_n_epochs,
        save_last=True,
        monitor="loss",
        mode="min",
        dirpath=f"{logger.log_dir}/models",
        filename="model_{step:07d}_{epoch:03d}",
        save_on_train_epoch_end=True,
    )

    checkpoint_callback_steps = ModelCheckpoint(
        save_top_k=-1,
        every_n_train_steps=cfg.experiment.chkpt_every_n_steps,
        save_last=True,
        monitor="loss",
        mode="min",
        dirpath=f"{logger.log_dir}/models",
        filename="model_{step:07d}_{epoch:03d}",
        save_on_train_epoch_end=True,
    )

    lr_monitor = LearningRateMonitor(logging_interval="step")

    print("Create trainer")
    trainer = pl.Trainer(
        logger=logger,
        callbacks=[checkpoint_callback, lr_monitor, checkpoint_callback_steps],
        **cfg.ztrainer,
    )

    print("Train " + str(cfg.experiment.run_name))

    trainer.fit(
        model=model,
        ckpt_path=cfg.experiment.resume_from_chkpt,
        datamodule=data_module,
    )
    print("Finished training")


if __name__ == "__main__":

    logging.basicConfig(
        format="%(asctime)s %(levelname)-8s %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    cfg = load_config()

    main(cfg)

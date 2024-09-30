import logging
from typing import Optional

import numpy as np
import pytorch_lightning as pl
import torch.utils.data as data

from hik.data.constants import activity2index

from sast.data.multi_person_data import MultiPersonData


def half_only_standing_mask(activities, primary_exists):

    is_only_standing = np.all(
        activities[..., activity2index["standing"]], axis=-1
    )  # (b p)

    logging.info(
        f"proportion to be kept before half masking {1.0-np.mean(is_only_standing.astype(np.float32))}"
    )

    half_mask = np.random.randint(0, 2, is_only_standing.shape, dtype="bool")

    is_interesting = np.logical_or(np.logical_not(is_only_standing), half_mask)  # (b p)

    logging.info(
        f"proportion to be kept after half masking {np.mean(is_interesting.astype(np.float32))}"
    )

    return is_interesting


class UniformDataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters(cfg)

        self.cfg = cfg

        mask_func = (
            half_only_standing_mask
            if cfg.loader.data_mask == "half_only_standing_mask"
            else None
        )

        self.train = MultiPersonData.load_from_files(
            f"{cfg.loader.dataset_path}/{cfg.loader.dataset}",
            cfg.loader.dataset_parts,
            data_mask_func=mask_func,
        )

    def calculate_train_statistics(self):

        return {"mean": self.train.get_mean(), "std": self.train.get_std()}

    def num_samples(self):

        return len(self.train)

    def setup(self, stage: Optional[str] = None):

        if stage in ["test"]:

            raise ValueError("currently unsupported")

    def train_dataloader(self):

        return data.DataLoader(
            self.train,
            batch_size=self.cfg.loader.batch_size,
            shuffle=self.cfg.loader.shuffle_train,
            num_workers=self.cfg.loader.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        raise ValueError("currently unsupported")

    def test_dataloader(self):
        raise ValueError("currently unsupported")

    def predict_dataloader(self):
        raise ValueError("currently unsupported")

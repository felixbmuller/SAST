import logging
import operator
import warnings

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from diffusers import DDPMScheduler
from diffusers.optimization import get_cosine_schedule_with_warmup
from einops import rearrange, repeat
from tqdm import tqdm
from yacs.config import CfgNode

from sast.model.unet import MultiUNet

logger = logging.getLogger("pytorch_lightning")


class MultiTcnDiffusion(pl.LightningModule):
    def __init__(self, cfg, data_mean=None, data_std=None):
        super().__init__()

        warnings.filterwarnings(
            "ignore", lineno=276
        )  # supress pytorch user warning about nested tensors
        # warnings.filterwarnings("ignore", lineno=172) # supress pytorch user warning about read-only tensors

        self.save_hyperparameters(cfg, ignore=["data_mean", "data_std"])

        self.cfg = cfg

        if isinstance(cfg, dict):
            self.cfg = cfg = CfgNode(cfg)

        if data_std is not None:
            data_std = np.where(data_std == 0.0, 1.0, data_std)

        self.n_primary_joints = cfg.data.n_joints

        self.register_buffer(
            "data_mean",
            (
                rearrange(
                    (
                        torch.from_numpy(data_mean)
                        if isinstance(data_mean, np.ndarray)
                        else data_mean
                    ),
                    "j d -> 1 1 j d",
                )
                if data_mean is not None
                else torch.empty((1, 1, self.n_primary_joints, 3))
            ),
        )
        self.register_buffer(
            "data_std",
            (
                rearrange(
                    (
                        torch.from_numpy(data_std)
                        if isinstance(data_std, np.ndarray)
                        else data_std
                    ),
                    "j d -> 1 1 j d",
                )
                if data_std is not None
                else torch.empty((1, 1, self.n_primary_joints, 3))
            ),
        )

        self.model = MultiUNet(cfg)

        # convert params from 0.8 to 0.14
        params = dict(**cfg.diffusion)
        params["prediction_type"] = "sample"

        self.noise_scheduler = DDPMScheduler(**params)

    def bootstrap_others(
        self, others, others_exists, target_shape, fake_single_person=False
    ):
        """
        Bootstrap others and others_exists by adding a zero-velocity continuation of the
        input sequences

        Returns
        -------
        others : Tensor(b p o t j d)
        """

        others_padded = torch.zeros(
            target_shape, dtype=torch.float32, device=self.device
        )

        if fake_single_person:
            return others_padded, torch.full_like(
                others_exists, False, dtype=torch.bool
            )

        others_padded[:, :, :, : others.shape[3]] = others
        others_padded[:, :, :, others.shape[3] :] = others[:, :, :, -1:]

        return others_padded, others_exists

    def infer_others(
        self, prev_pred, others_exists, mus, Rs, target_shape, fake_single_person=False
    ):
        """
        Infer others based on predictions in previous diffusion step

        Parameters
        ----------
        prev_pred : Tensor(b p t j d)
        others_exists : Tensor(b p o)
        mus : Tensor(b p 3)
        Rs : Tensor(b p 3 3)

        Returns
        -------
        others : Tensor(b p o t j d)
        others_exists : Tensor(b p o)
        """

        n_batch, n_persons, n_frames, n_prim_joints, n_dim = prev_pred.shape

        if fake_single_person:
            others_padded = torch.zeros(
                target_shape, dtype=torch.float32, device=self.device
            )

            return others_padded, torch.full_like(
                others_exists, False, dtype=torch.bool
            )

        # Undo standardization
        prev_pred = prev_pred * self.data_std + self.data_mean

        # Undo normalization
        Rs_T = torch.transpose(Rs, -1, -2)
        prev_pred = prev_pred @ rearrange(Rs_T, "b p d1 d2 -> b p 1 d1 d2")

        prev_pred = prev_pred + rearrange(mus, "b p d -> b p 1 1 d")

        # Apply normalization for everyone

        # prev_pred[b, p] contains all persons normalized using p's normalization
        prev_pred = repeat(prev_pred, "b p t j d -> b p p2 t j d", p2=n_persons)

        prev_pred = prev_pred - rearrange(mus, "b p d -> b p 1 1 1 d")
        prev_pred = prev_pred @ rearrange(Rs, "b p d1 d2 -> b p 1 1 d1 d2")

        # Remove current person from data

        selector = torch.logical_not(torch.eye(n_persons, device=self.device).bool())

        others_padded = rearrange(
            prev_pred[:, selector], "b (p o) t j d -> b p o t j d", p=n_persons
        )

        assert (
            others_padded.shape == target_shape
        ), f"wrong shapes {others_padded.shape=}, {target_shape}"

        return others_padded, others_exists

    @torch.no_grad()
    def forward(
        self,
        batch: dict,
        truncate_pred_seq=True,
        progress_bar=False,
        n_out=None,
        clip_range=None,
        noisy_in_seq=False,
    ):
        """Run a batched forward pass through the SoMoFormer model.

        Cave
        ----
        Note that the input sequences only contain frames_in and not the
        whole sequence! This also takes a primary person dimension, unlike the training loop.

        Parameters
        ----------
        batch : dict
            primary : Tensor(batch persons n_in joints 3)
            others: Tensor(b p o n_in j d)
            others_exists: BoolTensor(b p o)
                True if the other person is partially or fully present
            objects : Tensor(batch persons objects embed)
            norm_mus : Tensor(batch persons 3)
            norms_R : Tensor(batch persons 3 3)
        truncate_pred_seq : bool
            whether to remove the input sequence from the prediction before returning it

        Returns
        -------
        pred_seq: Tensor(b p t j d)
            final output
        """

        # GATHER INPUT DATA

        (
            primary,
            others,
            others_exists,
            objects,
            norm_mus,
            norm_Rs,
        ) = operator.itemgetter(
            "primary", "others", "others_exists", "objects", "mus", "Rs"
        )(
            batch
        )

        # COLLECT SHAPES

        if n_out is None:
            n_out = self.cfg.data.frames_out

        n_in = self.cfg.data.frames_in

        n_batch, n_persons = primary.shape[:2]

        n_frames = n_out + n_in

        others_shape = (
            n_batch,
            n_persons,
            n_persons - 1,
            n_frames,
            self.cfg.data.n_joints,
            3,
        )

        objects = rearrange(objects, "b p o e -> (b p) o e")

        if clip_range is None:
            self.noise_scheduler.config["clip_sample"] = False
        else:
            self.noise_scheduler.config["clip_sample"] = True
            self.noise_scheduler.config["clip_sample_range"] = clip_range

        assert primary.shape[2] == n_in, str(primary.shape)

        # STANDARDIZE PRIMARY INPUT

        primary = (primary - self.data_mean) / self.data_std

        sample = torch.normal(
            mean=0.0,
            std=1.0,
            size=(
                n_batch,
                n_persons,
                n_frames,
                self.n_primary_joints,
                3,
            ),
            device=self.device,
            dtype=torch.float32,
        )

        # BOOTSTRAP OTHER PERSONS

        # if True:
        # bootstrap using zero velocity
        step_others, step_others_exists = self.bootstrap_others(
            others,
            others_exists,
            others_shape,
            fake_single_person=False,
        )

        # PREDICT USING DIFFUSION LOOP

        iter = tqdm(self.noise_scheduler.timesteps) if progress_bar else self.noise_scheduler.timesteps  # type: ignore
        for i, t in enumerate(iter):

            timestep = torch.full(
                (n_batch * n_persons,), t, dtype=torch.long, device=self.device
            )

            step_sample = rearrange(sample, "b p t j d -> (b p) t j d")

            step_primary = rearrange(primary, "b p t j d -> (b p) t j d")

            if noisy_in_seq:

                input_noise = torch.normal(
                    mean=0.0,
                    std=1.0,
                    size=step_primary.shape,
                    device=self.device,
                    dtype=torch.float32,
                )

                noisy_input_seq = self.noise_scheduler.add_noise(
                    step_primary, input_noise, timestep
                )

                # Insert input sequence
                step_sample[:, :n_in] = noisy_input_seq

            else:

                step_sample[:, :n_in] = step_primary

            in_seq_padded = F.pad(step_primary, (0, 0, 0, 0, 0, n_out), "replicate")

            # concat in joint dimension
            step_sample = torch.cat([step_sample, in_seq_padded], dim=2)

            # Predict

            model_out = self.model(
                step_sample,
                rearrange(step_others, "b p o t j d -> (b p) o t j d"),
                rearrange(step_others_exists, "b p o -> (b p) o"),
                timestep,
                objects,
            )

            model_out = model_out[:, :, : self.n_primary_joints]

            model_out = rearrange(model_out, "(b p) t j d -> b p t j d", p=n_persons)

            if not getattr(self.cfg.unet, "remove_others", False):

                # Update others
                step_others, step_others_exists = self.infer_others(
                    model_out,
                    others_exists,
                    norm_mus,
                    norm_Rs,
                    others_shape,
                    fake_single_person=False,
                )

            # Update diffusion
            sample = self.noise_scheduler.step(model_out, t, sample).prev_sample  # type: ignore

        # UN-STANDARDIZE AND TRUNCATE

        out_seq = sample * self.data_std + self.data_mean

        if truncate_pred_seq:
            out_seq = out_seq[:, :, primary.shape[2] :]

        return {"pred_seq": out_seq}

    def training_step(self, batch, batch_idx):
        """Forward as used by train and validation step

        Parameters
        ----------
        batch : Dict
            primary: Tensor(b t j d)
            primary_exists: Tensor(b t)
                1.0 if the person is present 0.0 else
            others: Tensor(b o t j d)
                The primary person is not present here, i.e. o=p-1
            others_exists: BoolTensor(b o)
                True if the other person is partially or fully present
            objects : Tensor(b o e)
                embedded objects for primary person
        """

        primary, primary_exists, others, others_exists, objects = operator.itemgetter(
            "primary", "primary_exists", "others", "others_exists", "objects"
        )(batch)

        untransformed_primary = primary

        primary = (primary - self.data_mean) / self.data_std

        noise = torch.normal(
            mean=0.0,
            std=1.0,
            size=primary.shape,
            device=self.device,
            dtype=torch.float32,
        )

        timesteps = torch.randint(
            0,
            self.cfg.diffusion.num_train_timesteps,
            (primary.shape[0],),
            device=self.device,
            dtype=torch.int64,
        )  # (b)

        noisy_poses = self.noise_scheduler.add_noise(primary, noise, timesteps)  # type: ignore

        in_seq = primary[:, : self.cfg.data.frames_in]

        n_out = primary.shape[1] - self.cfg.data.frames_in
        in_seq_padded = F.pad(in_seq, (0, 0, 0, 0, 0, n_out), "replicate")

        # concat in joint dimension
        noisy_poses = torch.cat([noisy_poses, in_seq_padded], dim=2)

        model_out = self.model.forward(
            noisy_poses,
            others,
            others_exists,
            timesteps,
            objects,
        )

        model_out = model_out[:, :, : self.n_primary_joints]

        model_out = model_out * self.data_std + self.data_mean

        comparison_base = untransformed_primary

        model_out = model_out[:, self.cfg.data.frames_in :]
        comparison_base = comparison_base[:, self.cfg.data.frames_in :]
        primary_exists = primary_exists[:, self.cfg.data.frames_in :]

        loss = self.loss(model_out, comparison_base, primary_exists)

        self.log("loss", loss)

        return loss

    def loss(self, actual, expected, exists):
        """calculate mean loss, weighted by pose types

        Parameters
        ----------
        actual : Tensor(b t j d)
            model out
        expected : Tensor(b t j d)
            ground truth
        exists : Tensor(b t)
            1.0 if the person was actually in the scene in this moment, else 0.0
        """

        all_losses = torch.nn.functional.l1_loss(actual, expected, reduction="none")

        mean_losses = torch.mean(all_losses, dim=(-1, -2))  # (b t)

        mean_losses = mean_losses * exists

        loss = torch.mean(mean_losses)

        assert not torch.isnan(loss).any()

        return loss

    def configure_optimizers(self):

        optim = torch.optim.AdamW(self.model.parameters(), self.cfg.optim.lrate)

        lr_scheduler = get_cosine_schedule_with_warmup(
            optim,
            self.cfg.optim.warmup_steps,
            self.cfg.ztrainer.max_steps,
        )

        if self.cfg.optim.per_step:
            return [optim], [{"scheduler": lr_scheduler, "interval": "step"}]
        else:
            return {"optimizer": optim, "lr_scheduler": lr_scheduler}

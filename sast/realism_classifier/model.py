from typing import List
import numpy as np
from pytest import approx

import torch
from torch import nn
from torch.nn import functional as F
from einops.layers.torch import Rearrange
from fire import Fire
from tqdm import tqdm
from yacs.config import CfgNode
from einops import rearrange

from sast.model.modules.dct import dct
from sast.realism_classifier.config import get_cfg_defaults

def batch_normalize(seq, frame: int):
    """_summary_

    Parameters
    ----------
    seq : Tensor(batch frame joint dim)
        _description_
    frame : int
        _description_
    """

    #print("DIMS", str(seq.shape))

    # hip joints in hik's 29-joint layout (matches hik.transforms normalize jid_left/right)
    left3d = seq[:, frame, 1]  # b d
    right3d = seq[:, frame, 2]  # b d

    mu = (left3d + right3d) / 2  # batch dim
    mu[:, 2] = 0

    left2d = left3d[:, :2]
    right2d = right3d[:, :2]

    y = right2d - left2d  # b 2
    y = y / (torch.linalg.vector_norm(y, dim=-1, keepdim=True) + 0.00000001)

    angle = torch.arctan2(y[:, 1], y[:, 0])

    Rz = torch.stack([
        torch.stack([torch.cos(angle), -torch.sin(angle), torch.zeros_like(angle)]),
        torch.stack([torch.sin(angle), torch.cos(angle), torch.zeros_like(angle)]),
        torch.stack([torch.zeros_like(angle), torch.zeros_like(angle), torch.ones_like(angle)])
    ]) # dim dim batch

    Rz = rearrange(Rz, "d1 d2 b -> b 1 1 d1 d2")

    mu = rearrange(mu, "b d -> b 1 1 d")

    seq = seq - mu

    seq = rearrange(seq, "b t j d -> b t j 1 d")

    seq = seq @ Rz

    seq = rearrange(seq, "b t j 1 d -> b t j d")
    
    return seq


def test_batch_normalize():

    from hik.transforms.transforms import normalize

    seq = torch.randn(5, 250, 29, 3, dtype=torch.float32)

    seq[..., 2] = torch.abs(seq[..., 2])
    seq[..., :2] = seq[..., :2] * 10

    norm_seq = batch_normalize(seq, 0)

    reference_seq_ = []

    for b in range(5):
        # batch_normalize follows hik's 29-joint normalization (hips at 1/2).
        ref_seq = normalize(seq[b].numpy(), 0)

        reference_seq_.append(ref_seq)

    ref_seq = np.stack(reference_seq_)

    ref_seq = torch.from_numpy(ref_seq)

    assert norm_seq == approx(ref_seq, abs=1e-5)



class RealismClassifier(nn.Module):
    def __init__(self, cfg: CfgNode):
        super().__init__()

        self.cfg = cfg

        self.model = nn.Sequential(
            Rearrange("b (j d) t -> b j (d t)", d=3),
            nn.Linear(3*cfg.n_frames, cfg.n_hidden_temporal),
            nn.ReLU(),
            Rearrange("b j e -> b (j e)"),
            nn.Linear(cfg.n_hidden_temporal*cfg.n_joints, cfg.n_hidden_final),
            nn.ReLU(),
            nn.Linear(cfg.n_hidden_final, 1),
            nn.Sigmoid()
        )

    @classmethod
    def load(cls, save_path, device="cpu"):
        save = torch.load(save_path, map_location=device)

        model = cls(save["cfg"])

        model.load_state_dict(save["model_state"])

        return model.to(device)


    def forward(self, x):
        """Forward pass

        Parameters
        ----------
        x : Tensor(b t j d)
            input sequence

        Returns
        -------
        Tensor(b)
            classification between 0 (synthetic) and 1 (real)
        """

        x = batch_normalize(x, 0)

        x = rearrange(x, "b t j d -> b (j d) t")

        x = dct(x)

        x = self.model(x)

        x = rearrange(x, "b 1 -> b")

        return x
    
    @torch.no_grad()
    def forward_seq(self, batch, stepsize=5, device="cuda"):
        """Process a list of sequences of arbitrary length, by splitting it into overlapping 
        subsequences. This skips subsequences for which mask is False in at least one frame.

        Parameters
        ----------
        batch : List[Tuple[Tensor(t j d), Tensor(t)]]
            list of items, each of which is a sequence and a mask
        
        Returns
        -------
        List[Tensor(b)]
            One classification for each short sequence that batch was splitted into. This has the 
            same length as the input
        """

        pred = []

        for seq, mask in tqdm(batch):

            words = []

            zero_out = []

            for iframe in range(0, seq.shape[0] - self.cfg.n_frames + 1, stepsize):
                if not mask[iframe : iframe + self.cfg.n_frames].all():
                    zero_out.append(True)
                else:
                    zero_out.append(False)

                words.append(torch.from_numpy(seq[iframe : iframe + self.cfg.n_frames]))

            word_batch = torch.stack(words).to(torch.float32)

            y = self.forward(word_batch.to(device)).to("cpu")

            y[zero_out] = torch.nan

            pred.append(y)

        return pred

    def loss(self, actual, expected):
        return F.binary_cross_entropy(actual, expected, reduction="mean")

    def training_step(self, batch, batch_idx):
        X, y = batch

        return self.loss(self.forward(X), y)

    def configure_optimizers(self):
        bias_params = []
        other_params = []

        for name, param in self.named_parameters():
            if "bias" in name:
                bias_params.append(param)
            else:
                other_params.append(param)

        optim = torch.optim.AdamW(
            [{"params": other_params}, {"params": bias_params, "weight_decay": 0.0}],
            self.cfg.lrate,
        )

        return optim, None


def visualize_model(
    depth=3,
):
    from torchinfo import summary

    # print(channels)
    # print(time_embed_dim)

    cfg = get_cfg_defaults()

    model = RealismClassifier(cfg)

    summary(
        model,
        # input_size=[(cfg.loader.batch_size, 29 * 3, 256), (cfg.loader.batch_size,)],
        row_settings=["var_names"],
        depth=depth,
    )


if __name__ == "__main__":
    Fire(visualize_model)

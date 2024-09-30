import torch
from torch import nn
from fire import Fire
from einops import rearrange, repeat

from sast.modules.timestep import TimestepModule
from sast.model.scene_transformer import SceneTransformer
from sast.data.constants import object_embed_dim
from sast.modules.unet_parts import TcnDecoder, TcnEncoder
from sast.model.person_transformer import PersonTransformer


class MultiUNet(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        max_seq_len = 256

        is_causal = getattr(cfg.unet, "causal", True)

        # Timestep embedding
        self.timestep_module = TimestepModule(
            cfg.unet.time_embed_dim,
            time_embedding_type=getattr(cfg.unet, "time_embedding_type", "fourier"),
            freq_shift=0.0,
            flip_sin_to_cos=False,
            act_fn=None,
            use_timestep_embedding=False,
        )

        d_objects = cfg.unet.scene_transformer.d_model

        self.primary_enc = TcnEncoder(
            time_embed_dim=cfg.unet.time_embed_dim,
            d_objects=d_objects,
            is_causal=is_causal,
            **cfg.unet.primary_tcn
        )

        self.primary_dec = TcnDecoder(
            time_embed_dim=cfg.unet.time_embed_dim,
            d_objects=d_objects,
            is_causal=is_causal,
            **cfg.unet.primary_tcn
        )

        self.others_enc = TcnEncoder(
            time_embed_dim=cfg.unet.time_embed_dim,
            is_causal=is_causal,
            d_objects=d_objects,
            **cfg.unet.others_tcn
        )

        bottleneck_channels_primary = cfg.unet.primary_tcn.channels[-1]
        bottleneck_channels_others = cfg.unet.others_tcn.channels[-1]

        self.scene_transformer = SceneTransformer(
            dict(**cfg.unet.scene_transformer),
            bottleneck_channels_primary,
            object_embed_dim,
            max_seq_len,
            is_causal=is_causal,
            mask_memory_key=cfg.unet.mask_memory_key,
            remove_scene=getattr(cfg.unet, "remove_scene", False),
        )

        self.person_transformer = PersonTransformer(
            dict(**cfg.unet.person_transformer),
            bottleneck_channels_primary,
            bottleneck_channels_others,
            max_seq_len,
            is_causal=is_causal,
            remove_others=getattr(cfg.unet, "remove_others", False),
        )

        self.upsample = nn.Upsample(scale_factor=2.0, mode="linear")  # 32 -> 64

    def forward(self, primary, others, others_exists, timestep, objects):
        """Forward pass

        Parameters
        ----------
        primary : Tensor(b t j d)
            normalized and standardized sequences for primary persons
        others : Tensor(b p t j d)
            normalized (w.r.t. primary person) sequences of other persons, excluding primary person!
        others_exists : BoolTensor(b p)
            True if the other person if partially or fully present, i.e. not padding
        timestep : Tensor(b)
            diffusion timesteps
        objects: Tensor(b o e)
            objects, embedded for primary persons
        """

        # Remove persons from others that are padding for every sequence in the batch
        others_exists_somewhere = torch.any(others_exists, dim=0)  # p

        others = others[:, others_exists_somewhere]
        others_exists = others_exists[:, others_exists_somewhere]

        n_batch, n_others, n_frames, n_joints, n_dim = others.shape

        # Embed timestep
        time_embed = self.timestep_module(timestep)

        primary = rearrange(primary, "b t j d -> b (j d) t")

        h_primary, skips = self.primary_enc(primary, time_embed)

        # Combine primaries and environment
        h_primary = self.scene_transformer(h_primary, objects)

        # Encode others
        if n_others > 0:
            # if all entries in batch are single-person, ignore others encoder
            h_others = rearrange(others, "b p t j d -> (b p) (j d) t")

            h_others, _ = self.others_enc(
                h_others, repeat(time_embed, "b e -> (b p) e", p=n_others)
            )
            h_others = rearrange(h_others, "(b p) c t -> b p c t", p=n_others)

        else:
            h_others = None

        # Combine primaries and others
        h_primary = self.person_transformer(
            h_primary, h_others, others_exists
        )  # (b c t)

        # Decode primary
        h_primary = self.upsample(h_primary)
        h_primary = self.primary_dec(h_primary, time_embed, skips)  # (bp jd t)

        out = rearrange(
            h_primary,
            "b (j d) t -> b t j d",
            d=3,
        )

        return out


if __name__ == "__main__":
    Fire(visualize_model)

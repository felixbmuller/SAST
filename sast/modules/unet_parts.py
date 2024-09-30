import torch
from torch import nn

from sast.modules.conv_block import TemporalBlock


class TcnEncoder(nn.Module):
    def __init__(
        self,
        channels,
        time_embed_dim,
        kernel_size,
        dropout,
        norm_mode,
        use_residual,
        d_objects,
        group_norm_groups,
        is_causal,
        timestep_mode="add",
        padding_mode="zero",
    ):
        super().__init__()

        use_norm = norm_mode == "group_norm"

        down_cfg = dict(
            n_time_embed=time_embed_dim,
            kernel_size=kernel_size,
            dropout=dropout,
            use_norm=use_norm,
            use_residual=use_residual,
            group_norm_groups=group_norm_groups,
            d_objects=d_objects,  # output dim of transformer
            stride=2,
            is_causal=is_causal,
            timestep_mode=timestep_mode,
            padding_mode=padding_mode,
        )

        self.down = nn.ModuleList()
        for i in range(1, len(channels)):
            self.down.append(TemporalBlock(channels[i - 1], channels[i], **down_cfg))
            # every block halfs the number of frames

    def forward(self, sample, time_embed):
        """Forward pass

        Parameters
        ----------
        sample : Tensor(b (j d) t)
            input sequence
        time_embed : Tensor(b e)
        """

        down_out = sample
        down_skip = []
        for layer in self.down:
            down_out, skip = layer(down_out, time_embed)
            down_skip.append(skip)

        return down_out, down_skip


class TcnDecoder(nn.Module):
    def __init__(
        self,
        channels,
        time_embed_dim,
        kernel_size,
        dropout,
        norm_mode,
        use_residual,
        d_objects,
        group_norm_groups,
        is_causal,
        timestep_mode="add",
        padding_mode="zero",
    ):
        """
        channels:
            channels for the decoder, including output channels
        """
        super().__init__()

        use_norm = norm_mode == "group_norm"

        channels = list(reversed(channels))

        general_cfg = dict(
            n_time_embed=time_embed_dim,
            kernel_size=kernel_size,
            dropout=dropout,
            use_norm=use_norm,
            use_residual=use_residual,
            group_norm_groups=group_norm_groups,
            d_objects=d_objects,  # output dim of transformer
            is_causal=is_causal,
            timestep_mode=timestep_mode,
            padding_mode=padding_mode,
        )

        up_cfg = dict(
            **general_cfg,
            upsample_mode="linear",
            upsample_factor=2.0,
        )

        self.up = nn.ModuleList()

        for i in range(1, len(channels) - 1):
            self.up.append(
                TemporalBlock(
                    2 * channels[i - 1], channels[i], n_hidden=channels[i - 1], **up_cfg
                )
            )
            # each upsample layer doubles the number of frames

        self.out = TemporalBlock(
            2 * channels[-2],
            channels[-1],
            n_hidden=channels[-2],
            last_block=True,
            **general_cfg,
        )  # 256 -> 256

    def forward(self, x, time_embed, down_skip, obj_enc=None, obj_enc_mask=None):
        """Forward pass

        Parameters
        ----------
        x : Tensor(b (j d) t)
            input sequence
        time_embed : Tensor(b e)
        down_skip: Tensor(s b c t)
        """

        up_out = x

        for layer in self.up:
            up_out = layer(
                torch.cat((down_skip.pop(), up_out), dim=1),
                time_embed,
                obj_enc=obj_enc,
                obj_enc_mask=obj_enc_mask,
            )[0]

        up_out = self.out(
            torch.cat((down_skip.pop(), up_out), dim=1),
            time_embed,
            obj_enc=obj_enc,
            obj_enc_mask=obj_enc_mask,
        )[0]

        return up_out

import logging

import torch
from einops import rearrange, repeat
from torch import nn
from torch.nn.utils.weight_norm import weight_norm


class CausalConv1dBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        dropout,
        stride,
        act_fn,
        last_block,
        use_norm,
        group_norm_groups,
        is_causal,
        d_objects=None,
        padding_mode="zero",
    ):
        super().__init__()

        padding = kernel_size - 1

        self.last_block = last_block

        if is_causal:
            self.pad = torch.nn.ConstantPad1d((padding, 0), value=0.0)
        else:
            if padding_mode == "zero":
                self.pad = torch.nn.ConstantPad1d(
                    (padding // 2, padding // 2), value=0.0
                )
            elif padding_mode == "replicate":
                self.pad = torch.nn.ReplicationPad1d((padding // 2, padding // 2))
            else:
                ValueError(str(padding_mode))
        self.dropout = nn.Dropout(dropout) if not last_block else nn.Identity()
        self.act = act_fn()

        self.conv = weight_norm(
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=0,
            )
        )

        if not use_norm or last_block:
            self.norm = nn.Identity()
        elif out_channels % group_norm_groups != 0:
            logging.warn(
                f"No group norm possible after convolution {in_channels}->{out_channels}"
            )
            self.norm = nn.Identity()
        elif use_norm == "group_norm":
            self.norm = nn.GroupNorm(group_norm_groups, out_channels)
        else:
            assert False

        if last_block:
            self.model = nn.Sequential(
                self.pad,
                self.conv,
            )
        else:
            self.model = nn.Sequential(
                self.pad,
                self.conv,
                self.act,
                self.norm,
                self.dropout,
            )

        self.conv.weight.data.normal_(0, 0.01)

    def forward(self, x, obj_enc=None, obj_enc_mask=None):
        """Perform causal convolution

        Parameters
        ----------
        x : Tensor(b in_channels t)
            input sequence

        Returns
        -------
        Tensor(b out_channels t_out)
            output sequence with t_out = floor((t - 1)/stride +1)
            TODO: Why this formula?
        """

        return self.model(x)


class TemporalBlock(nn.Module):
    """from https://www.kaggle.com/code/ceshine/pytorch-temporal-convolutional-networks/script"""

    def __init__(
        self,
        n_inputs,
        n_outputs,
        n_time_embed,
        kernel_size,
        dropout,
        stride=1,
        act_fn=nn.ReLU,
        n_hidden=None,
        upsample_factor=1.0,
        upsample_mode=None,
        group_norm_groups=25,
        last_block=False,
        use_norm=False,
        use_residual=True,
        d_objects=None,
        is_causal=True,
        timestep_mode="add",
        padding_mode="zero",
    ):
        super(TemporalBlock, self).__init__()

        if n_hidden is None:
            n_hidden = n_outputs

        self.group_norm_groups = group_norm_groups
        self.use_residual = use_residual
        self.timestep_mode = timestep_mode

        n_timestep_dim = n_hidden if timestep_mode == "add" else 128

        # Time
        self.time_emb_act = nn.Mish()
        if timestep_mode == "none":
            self.time_emb = nn.Identity()
        else:
            self.time_emb = nn.Linear(n_time_embed, n_timestep_dim)

        use_norm_str = False
        if use_norm:
            use_norm_str = "group_norm"

        self.conv1 = CausalConv1dBlock(
            in_channels=n_inputs,
            out_channels=n_hidden,
            kernel_size=kernel_size,
            dropout=dropout,
            stride=1,
            act_fn=act_fn,
            last_block=False,
            use_norm=use_norm_str,
            group_norm_groups=group_norm_groups,
            is_causal=is_causal,
            d_objects=d_objects,
            padding_mode=padding_mode,
        )

        self.conv2 = CausalConv1dBlock(
            in_channels=(
                n_hidden + n_timestep_dim if timestep_mode == "concat" else n_hidden
            ),
            out_channels=n_outputs,
            kernel_size=kernel_size,
            dropout=dropout,
            stride=stride,
            act_fn=act_fn,
            last_block=last_block,
            use_norm=use_norm_str,
            group_norm_groups=group_norm_groups,
            is_causal=is_causal,
            d_objects=d_objects,
            padding_mode=padding_mode,
        )

        if use_residual:

            self.resize_residual = nn.Conv1d(n_inputs, n_outputs, 1, stride=stride)
            self.resize_residual.weight.data.normal_(0, 0.01)

        if upsample_mode is not None:
            self.upsample = nn.Upsample(
                scale_factor=upsample_factor, mode=upsample_mode
            )
        else:
            self.upsample = nn.Identity()

    def forward(self, x, t, p=None, obj_enc=None, obj_enc_mask=None):
        """
        Parameters
        ----------
        x : Tensor(b in_channels t)
            sample to process
        t : Tensor(b embed_dim)
            embedded timestep
        p : Tensor(b pose_type_channels t)

        Returns
        -------
        Tensor(b out_channels t)

        """
        t = self.time_emb_act(t)
        t = self.time_emb(t)
        t = rearrange(t, "b out_c -> b out_c 1")

        out = x
        out = self.conv1(out, obj_enc=obj_enc, obj_enc_mask=obj_enc_mask)

        skip = out

        if self.timestep_mode == "concat":
            t_large = repeat(t, "b out_c 1 -> b out_c ts", ts=x.shape[-1])
            out = torch.cat([out, t_large], dim=1)
        elif self.timestep_mode == "add":
            out = out + t
        elif self.timestep_mode == "none":
            pass
        else:
            ValueError("illegal timestep_mode")

        out = self.conv2(out, obj_enc=obj_enc, obj_enc_mask=obj_enc_mask)

        if self.use_residual:
            res = self.resize_residual(x)
            out = out + res

        out = self.upsample(out)

        return out, skip

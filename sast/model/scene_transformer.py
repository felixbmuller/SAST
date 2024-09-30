import torch
from torch import nn
from einops import rearrange

from sast.modules.positional_encoding import PositionalEncoding


class SceneTransformer(nn.Module):
    def __init__(
        self,
        transformer_cfg,
        n_channels,
        d_obj_embed,
        max_seq_len,
        is_causal,
        mask_memory_key=False,
        remove_scene=False,
    ) -> None:
        super().__init__()

        self.is_causal = is_causal
        self.remove_scene = remove_scene
        self.d_model = transformer_cfg["d_model"]
        self.mask_memory_key = mask_memory_key

        self.map_objs = nn.Linear(d_obj_embed, self.d_model)
        self.map_channels = nn.Linear(n_channels, self.d_model)
        self.inv_map_channels = nn.Linear(self.d_model, n_channels)

        self.dropout = nn.Dropout(transformer_cfg["dropout"])

        self.transformer = nn.Transformer(batch_first=True, **transformer_cfg)

        self.positional_encoding = PositionalEncoding(self.d_model, max_seq_len)

    def forward_encoder(self, objects):

        # sum for all non-padding objects is >= 1 because of the one_hot label encoding
        objects_mask = torch.sum(objects, dim=-1) < 0.1  # (b o)

        objects = self.map_objs(objects)
        objects = self.dropout(objects)  # (b o d)

        memory = self.transformer.encoder(
            objects, mask=None, src_key_padding_mask=objects_mask
        )

        memory_key_mask = (
            objects_mask[:, : memory.shape[1]] if self.mask_memory_key else None
        )

        return memory, memory_key_mask

    def forward(self, hidden, objects, return_hidden=False):
        """combine the encoder state with the scene embedding

        Parameters
        ----------
        hidden : Tensor(b c t)
            encoder hidden state, downsamples number of frames
        objects : Tensor(b o e)
            scene embedding

        Returns
        -------
        Tensor(b c t) TODO really same shape?
        """

        hidden = rearrange(hidden, "b c t -> b t c")
        hidden = self.map_channels(hidden)
        hidden = self.positional_encoding(hidden)
        hidden = self.dropout(hidden)  # (b t d)

        # apply transformer

        if self.is_causal:

            temporal_mask = nn.Transformer.generate_square_subsequent_mask(
                hidden.shape[1], device=hidden.device
            )

        else:
            temporal_mask = None

        if self.remove_scene:
            memory = hidden.new_zeros(objects.shape[0], 1, self.d_model)
            memory_key_mask = None
        else:
            memory, memory_key_mask = self.forward_encoder(objects)

        out = self.transformer.decoder(
            hidden,
            memory,
            tgt_mask=temporal_mask,
            memory_mask=None,
            tgt_key_padding_mask=None,
            memory_key_padding_mask=memory_key_mask,
        )

        out = self.inv_map_channels(out)

        out = rearrange(out, "b t c -> b c t")

        if return_hidden:
            return out, memory, memory_key_mask
        else:
            return out

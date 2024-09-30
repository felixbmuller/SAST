import torch
from torch import Tensor, nn
from einops import rearrange, repeat

from sast.modules.positional_encoding import PositionalEncoding


class PersonTransformer(nn.Module):
    def __init__(
        self,
        transformer_cfg,
        n_channels_primary,
        n_channels_others,
        max_seq_len,
        is_causal,
        remove_others=False,
    ) -> None:
        super().__init__()

        self.is_causal = is_causal
        self.remove_others = remove_others
        self.d_model = transformer_cfg["d_model"]

        self.map_primary = nn.Linear(n_channels_primary, self.d_model)
        self.map_others = nn.Linear(n_channels_others, self.d_model)
        self.inv_map_channels = nn.Linear(self.d_model, n_channels_primary)

        self.dropout = nn.Dropout(transformer_cfg["dropout"])

        self.transformer = nn.Transformer(batch_first=True, **transformer_cfg)

        self.positional_encoding = PositionalEncoding(self.d_model, max_seq_len)

    def forward_encoder(self, others, others_present):
        """
        Encode other persons.

        There is a lot of special case handling here, because sequences might be empty if the primary person is alone in the room. To avoid NaNs, this code ensures that the encoder is always fed non-empty sequences, but the output for empty sequences is zeroed out, which is as close to ignoring the encoder as it gets.

        Note that tokens can only attend to other tokens of the same timestep, i.e. for [A1, A2, B1, B2], A1 can attend to B1, but not A2 and B2. This is enforced via src_mask and memory_mask.
        """

        # Bypass encoder, just send nothing if there are no others in the whole batch
        if others is None:
            memory_key_mask = None
            memory_mask = None
            memory = torch.zeros(
                (others_present.shape[0], 1, self.d_model),
                device=others_present.device,
                dtype=torch.float32,
            )
            return memory, memory_mask, memory_key_mask

        n_frames = others.shape[-1]
        n_persons_other = others.shape[1]

        assert n_persons_other > 0, "Cannot deal with no persons"

        # Preprocess

        others = rearrange(others, "b p c t -> b (p t) c")
        others = self.map_others(others)
        others = self.dropout(others)  # (b pt c)

        neg_eye = torch.logical_not(
            torch.eye(n_frames, device=others.device, dtype=torch.bool)
        )

        any_others_present = rearrange(torch.any(others_present, dim=-1), "b -> b 1")

        # ENCODER

        # every src frame can only see the same frame of other persons
        src_mask = repeat(
            neg_eye, "t1 t2 -> (p1 t1) (p2 t2)", p1=n_persons_other, p2=n_persons_other
        )

        src_key_padding_mask = repeat(
            torch.logical_not(others_present), "b p -> b (p t)", t=n_frames
        )

        # avoid empty sequences in the encoder as it leads to NaNs
        src_key_padding_mask = torch.where(
            any_others_present, src_key_padding_mask, False
        )

        memory = self.transformer.encoder(
            others, mask=src_mask, src_key_padding_mask=src_key_padding_mask
        )  # b pt c

        # Handle alone scenes
        # For scenes without anyone else, the encoder produces Nan outputs
        # For alone scenes, we will pass memory=[0, ..., 0] into the decoder, allowing it to effectively
        # just ignore the input

        memory = torch.where(rearrange(any_others_present, "b 1 -> b 1 1"), memory, 0.0)

        # DECODER

        memory_mask = repeat(neg_eye, "t1 t2 -> t1 (p2 t2)", p2=n_persons_other)

        memory_key_mask = src_key_padding_mask[
            :, : memory.shape[1]
        ]  # Transformer sequeezes memory

        return memory, memory_mask, memory_key_mask

    def forward(self, primary: Tensor, others, others_present):
        """combine the encoder state with the scene embedding

        Parameters
        ----------
        primary : Tensor(b c t)
            encoder hidden state, downsamples number of frames
        others : Tensor(b p c t)
            scene embedding
        others_present : BoolTensor(b p)
            True if the corresponding person should be included

        Returns
        -------
        Tensor(b c t)
        """

        n_frames = primary.shape[-1]

        primary = rearrange(primary, "b c t -> b t c")
        primary = self.map_primary(primary)
        primary = self.positional_encoding(primary)
        primary = self.dropout(primary)  # (b t d)

        if self.remove_others:
            memory = primary.new_zeros(others.shape[0], 1, self.d_model)
            memory_key_mask = memory_mask = None
        else:
            memory, memory_mask, memory_key_mask = self.forward_encoder(
                others, others_present
            )

        if self.is_causal:

            tgt_mask = nn.Transformer.generate_square_subsequent_mask(
                n_frames, device=primary.device
            )

        else:
            tgt_mask = None

        out = self.transformer.decoder(
            primary,
            memory,
            tgt_mask=tgt_mask,
            memory_mask=memory_mask,
            tgt_key_padding_mask=None,
            memory_key_padding_mask=memory_key_mask,
        )

        out = self.inv_map_channels(out)

        out = rearrange(out, "b t c -> b c t")

        return out

import torch.nn as nn
from diffusers.models.embeddings import (
    GaussianFourierProjection,
    TimestepEmbedding,
    Timesteps,
)


class TimestepModule(nn.Module):
    def __init__(
        self,
        embed_dim,
        time_embedding_type,
        flip_sin_to_cos,
        freq_shift,
        use_timestep_embedding,
        act_fn,
    ):
        super().__init__()

        if time_embedding_type == "fourier":
            assert embed_dim % 2 == 0, "embedding_dim for fourier must be even"

            self.time_proj = GaussianFourierProjection(
                embedding_size=embed_dim // 2,
                set_W_to_weight=False,
                log=False,
                flip_sin_to_cos=flip_sin_to_cos,
            )  # b -> b embed_dim

        elif time_embedding_type == "positional":
            self.time_proj = Timesteps(
                embed_dim,
                flip_sin_to_cos=flip_sin_to_cos,
                downscale_freq_shift=freq_shift,
            )  # b -> b embed_dium
        else:
            assert False

        if use_timestep_embedding:
            time_embed_dim = embed_dim * 4
            self.time_mlp = TimestepEmbedding(
                in_channels=embed_dim,
                time_embed_dim=time_embed_dim,
                act_fn=act_fn,
                out_dim=embed_dim,
            )  # b embed_dim -> b embed_dim
        else:
            self.time_mlp = nn.Identity()

    def forward(self, timesteps):
        """

        Parameters
        ----------
        timesteps : LongTensor(b)
            timesteps

        Returns
        -------
        Tensor(b embed_dim)
        """

        timestep_embed = self.time_proj(timesteps)
        timestep_embed = self.time_mlp(timestep_embed)

        return timestep_embed

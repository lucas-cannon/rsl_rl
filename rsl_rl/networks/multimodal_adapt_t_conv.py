import torch
import torch.nn as nn
from rsl_rl.networks import CNN, ProprioAdaptTConv


class MultimodalAdaptTConv(nn.Module):

    def __init__(
        self,
        obs: dict,
        obs_hist_keys: list[str],
        history_len: int,
        extrinsics_output_dim: int,
        cnn_cfg: dict | None,
    ):
        super().__init__()

        self.history_len = history_len
        self.hist_1d_keys = []
        self.hist_2d_keys = []

        for key in obs_hist_keys:
            shape = obs[key].shape

            if len(shape) == 3:        # (B, T, D)
                assert shape[1] == history_len
                self.hist_1d_keys.append(key)

            elif len(shape) == 5:      # (B, T, C, H, W)
                assert shape[1] == history_len
                self.hist_2d_keys.append(key)

            else:
                raise ValueError(f"Unsupported obs_hist shape for {key}: {shape}")

        self.hist_cnns = nn.ModuleDict()
        self.hist_cnn_out_dim = {}

        if self.hist_2d_keys:
            assert cnn_cfg is not None, "cnn_cfg must be provided when 2D obs exist"

        for key in self.hist_2d_keys:
            _, T, C, H, W = obs[key].shape

            cnn = CNN(
                input_dim=(H, W),
                input_channels=C,
                **cnn_cfg,
            )

            self.hist_cnns[key] = cnn

            if cnn.output_channels is not None:
                raise ValueError("CNN output must be flattened")

            self.hist_cnn_out_dim[key] = int(cnn.output_dim)

        latent_dim = 0

        for key in self.hist_1d_keys:
            latent_dim += obs[key].shape[-1]

        for key in self.hist_2d_keys:
            latent_dim += self.hist_cnn_out_dim[key]

        self.temporal = ProprioAdaptTConv(
            proprio_input_dim=latent_dim,
            history_len=history_len,
            extrinsics_output_dim=extrinsics_output_dim,
        )

    def forward(self, obs: dict):

        latent_list = []

        for key in self.hist_1d_keys:
            latent_list.append(obs[key])   # (B, T, D)

        for key in self.hist_2d_keys:
            x = obs[key]  # (B, T, C, H, W)
            B, T, C, H, W = x.shape

            x = x.reshape(B * T, C, H, W)
            feat = self.hist_cnns[key](x)      # (B*T, F)
            feat = feat.reshape(B, T, -1)      # (B, T, F)

            latent_list.append(feat)

        if len(latent_list) == 1:
            z_seq = latent_list[0]
        else:
            z_seq = torch.cat(latent_list, dim=-1)

        return self.temporal(z_seq)

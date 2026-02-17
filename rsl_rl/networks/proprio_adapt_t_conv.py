import torch
import torch.nn as nn

class ProprioAdaptTConv(nn.Module):
    def __init__(
        self,
        proprio_input_dim: int,
        history_len: int,
        extrinsics_output_dim: int,
    ):
        super().__init__()

        # input expected shape: (B, T, D)
        # e.g. (4096, 50, 32)

        self.channel_transform = nn.Sequential(
            nn.Linear(proprio_input_dim, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 32),
            nn.ReLU(inplace=True),
        )

        self.temporal_aggregation = nn.Sequential(
            nn.Conv1d(32, 32, kernel_size=9, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 32, kernel_size=5, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 32, kernel_size=5, stride=1),
            nn.ReLU(inplace=True),
        )

        # compute final temporal size dynamically
        dummy = torch.zeros(1, history_len, proprio_input_dim)
        with torch.no_grad():
            d = self.channel_transform(dummy)
            d = d.permute(0, 2, 1)
            d = self.temporal_aggregation(d)
            final_dim = d.shape[-1]

        self.low_dim_proj = nn.Linear(32 * final_dim, extrinsics_output_dim)

    def forward(self, x):
        # x: (B, T, D)
        x = self.channel_transform(x)          # (B, T, 32)
        x = x.permute(0, 2, 1)                # (B, 32, T)
        x = self.temporal_aggregation(x)      # (B, 32, T')
        x = x.flatten(1)
        x = self.low_dim_proj(x)
        return torch.tanh(x)

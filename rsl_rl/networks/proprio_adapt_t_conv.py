import torch
import torch.nn as nn
from ipdb import set_trace

class ProprioAdaptTConv(nn.Module):
    def __init__(
        self,
        proprio_input_dim: int,     # e.g. 92
        history_len: int,           # e.g. 30
        extrinsics_output_dim: int, # e.g. 8
    ):
        super().__init__()

        hidden_units = proprio_input_dim  # match original RMA width

        # Channel transform (per time step)
        self.channel_transform = nn.Sequential(
            nn.Linear(proprio_input_dim, hidden_units),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_units, hidden_units),
            nn.ReLU(inplace=True),
        )

        # Temporal aggregation (Conv over time)
        # self.temporal_aggregation = nn.Sequential(
        #     nn.Conv1d(hidden_units, hidden_units, kernel_size=9, stride=2),
        #     nn.ReLU(inplace=True),
        #     nn.Conv1d(hidden_units, hidden_units, kernel_size=5, stride=1),
        #     nn.ReLU(inplace=True),
        #     nn.Conv1d(hidden_units, hidden_units, kernel_size=5, stride=1),
        #     nn.ReLU(inplace=True),
        # )

        if history_len <= 1:
            self.temporal_aggregation = nn.Identity()
            final_temporal_dim = 1
        elif history_len <= 10:
            self.temporal_aggregation = nn.Sequential(
                nn.Conv1d(hidden_units, hidden_units, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv1d(hidden_units, hidden_units, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv1d(hidden_units, hidden_units, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
            )
        elif history_len == 15:
            self.temporal_aggregation = nn.Sequential(
                nn.Conv1d(hidden_units, hidden_units, kernel_size=5, stride=2),
                nn.ReLU(inplace=True),
                nn.Conv1d(hidden_units, hidden_units, kernel_size=3, stride=1),
                nn.ReLU(inplace=True),
                nn.Conv1d(hidden_units, hidden_units, kernel_size=3, stride=1),
                nn.ReLU(inplace=True),
            )
        elif history_len == 30:
            self.temporal_aggregation = nn.Sequential(
                nn.Conv1d(hidden_units, hidden_units, kernel_size=9, stride=2),
                nn.ReLU(inplace=True),
                nn.Conv1d(hidden_units, hidden_units, kernel_size=5, stride=1),
                nn.ReLU(inplace=True),
                nn.Conv1d(hidden_units, hidden_units, kernel_size=5, stride=1),
                nn.ReLU(inplace=True),
            )
        else:
            raise ValueError(f"Unsupported history_len {history_len}. Supported: <=1, <=10, 30.")


        # Dynamically compute final temporal dimension
        with torch.no_grad():
            dummy = torch.zeros(1, history_len, proprio_input_dim)
            d = self.channel_transform(dummy)      # (1, T, H)
            d = d.permute(0, 2, 1)                # (1, H, T)
            d = self.temporal_aggregation(d)      # (1, H, T')
            final_temporal_dim = d.shape[-1]

        # Projection to low-dim extrinsics latent
        self.low_dim_proj = nn.Linear(
            hidden_units * final_temporal_dim,
            extrinsics_output_dim,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, D)
           B = batch size
           T = history length
           D = proprio_input_dim
        """

        x = self.channel_transform(x)        # (B, T, H)
        x = x.permute(0, 2, 1)              # (B, H, T)
        x = self.temporal_aggregation(x)    # (B, H, T')
        x = x.flatten(1)                    # (B, H*T')
        x = self.low_dim_proj(x)            # (B, extrinsics_output_dim)

        return torch.tanh(x)                # match RMA design
        # return x                # match RMA design

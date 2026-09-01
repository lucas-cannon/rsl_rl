# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn
from tensordict import TensorDict
from torch.distributions import Normal
from typing import Any, NoReturn
from rsl_rl.networks import MLP, EmpiricalNormalization, HiddenState, ProprioAdaptTConv

class StudentTeacherExtrinsics(nn.Module):
    is_recurrent: bool = False

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        num_actions: int,
        extrinsics_output_dim: int,
        history_len: int,
        actor_obs_normalization: bool = False,
        priv_obs_normalization: bool = False,
        student_1d_obs_normalization: bool = False,
        activation="elu",
        actor_hidden_dims=(256, 128),
        teacher_extrinsics_hidden_dims=(256, 128),
        init_noise_std: float = 0.1,
        noise_std_type: str = "scalar",
        # teacher_tanh_extrinsics: bool = False,
    ):
        super().__init__()

        self.obs_groups = obs_groups
        self.loaded_teacher = False
        self.extrinsics_output_dim = extrinsics_output_dim
        # self.use_student_extrinsics_for_rollout = True
        # -----------------------------
        # Priv dims (teacher)
        # -----------------------------
        priv_dim = sum(obs[k].shape[-1] for k in obs_groups["priv"])
        obs_dim = sum(obs[k].shape[-1] for k in obs_groups["obs"])
        
        obs_hist_key = obs_groups["obs_hist"][0]
        hist_shape = obs[obs_hist_key].shape  # (B, T, D)

        assert len(hist_shape) == 3, \
            f"Expected temporal proprio shape (B, T, D), got {hist_shape}"

        assert hist_shape[1] == history_len, \
            f"History length mismatch: expected {history_len}, got {hist_shape[1]}"

        proprio_input_dim = hist_shape[2]


        self.teacher_extrinsics_encoder = MLP(
            priv_dim,
            extrinsics_output_dim,
            teacher_extrinsics_hidden_dims,
            activation,
            # last_activation="tanh" if teacher_tanh_extrinsics else None,
        )
        self.student_extrinsics_encoder = ProprioAdaptTConv(
            proprio_input_dim=proprio_input_dim,
            history_len=history_len,
            extrinsics_output_dim=extrinsics_output_dim,
        )

        # Freeze teacher
        self.teacher_extrinsics_encoder.eval()
        for p in self.teacher_extrinsics_encoder.parameters():
            p.requires_grad = False

        self.history_len = history_len
        self.proprio_input_dim = proprio_input_dim

        actor_input_dim = obs_dim + extrinsics_output_dim

        self.actor = MLP(
            actor_input_dim,
            num_actions,
            actor_hidden_dims,
            activation,
        )

        # -------------------------------------------------
        # Freeze teacher + actor
        # -------------------------------------------------
        self.teacher_extrinsics_encoder.eval()
        for p in self.teacher_extrinsics_encoder.parameters():
            p.requires_grad = False

        self.actor.eval()
        for p in self.actor.parameters():
            p.requires_grad = False

        # normalization
        if actor_obs_normalization:
            self.obs_normalizer = EmpiricalNormalization(obs_dim)
        else:
            self.obs_normalizer = nn.Identity()
        if priv_obs_normalization:
            self.priv_normalizer = EmpiricalNormalization(priv_dim)
        else:
            self.priv_normalizer = nn.Identity()
        if student_1d_obs_normalization:
            self.obs_hist_normalizer = EmpiricalNormalization(
                history_len * proprio_input_dim
            )
        else:
            self.obs_hist_normalizer = nn.Identity()

        # Action noise (for rollout)
        self.noise_std_type = noise_std_type
        if self.noise_std_type == "scalar":
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        elif self.noise_std_type == "log":
            self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")

        self.distribution = None
        Normal.set_default_validate_args(False)

    def reset(
        self, dones: torch.Tensor | None = None, hidden_states: tuple[HiddenState, HiddenState] = (None, None)
    ) -> None:
        pass

    def forward(self) -> NoReturn:
        raise NotImplementedError

    @property
    def action_mean(self) -> torch.Tensor:
        return self.distribution.mean

    @property
    def action_std(self) -> torch.Tensor:
        return self.distribution.stddev

    @property
    def entropy(self) -> torch.Tensor:
        return self.distribution.entropy().sum(dim=-1)

    def get_student_extrinsics(self, obs: TensorDict) -> torch.Tensor:
        hist = torch.cat(
            [obs[k] for k in self.obs_groups["obs_hist"]],
            dim=-1,
        )  # (B, T, D_total)

        if isinstance(self.obs_hist_normalizer, EmpiricalNormalization):
            B, T, D = hist.shape
            hist_flat = hist.reshape(B, T * D)
            hist_flat = self.obs_hist_normalizer(hist_flat)
            hist = hist_flat.view(B, T, D)

        return self.student_extrinsics_encoder(hist)

    def get_actor_obs(self, obs: TensorDict):
        # if self.use_student_extrinsics_for_rollout:
        #     extrinsics_vec = self.get_student_extrinsics(obs)
        # else:
        #     extrinsics_vec = self.get_teacher_extrinsics(obs)
        #     print("testing using teacher extrinsics...")
        extrinsics_vec = self.get_student_extrinsics(obs)
        # print("s2:", extrinsics_vec.min().item(), extrinsics_vec.max().item())
        obs_vec = torch.cat([obs[k] for k in self.obs_groups["obs"]], dim=-1)
        return obs_vec, extrinsics_vec

    def act(self, obs: TensorDict, **kwargs) -> torch.Tensor:
        obs_vec, extrinsics_vec = self.get_actor_obs(obs)
        obs_vec = self.obs_normalizer(obs_vec)
        self._update_distribution(obs_vec, extrinsics_vec)
        return self.distribution.sample()

    def act_inference(self, obs: TensorDict) -> torch.Tensor:
        obs_vec, extrinsics_vec = self.get_actor_obs(obs)
        obs_vec = self.obs_normalizer(obs_vec)
        mlp_obs = torch.cat([obs_vec, extrinsics_vec], dim=-1)
        return self.actor(mlp_obs)

    def _update_distribution(self, obs_vec: torch.Tensor, extrinsics_vec: torch.Tensor) -> None:
        mlp_obs = torch.cat([obs_vec, extrinsics_vec], dim=-1)
        mean = self.actor(mlp_obs)
        if self.noise_std_type == "scalar":
            std = self.std.expand_as(mean)
        elif self.noise_std_type == "log":
            std = torch.exp(self.log_std).expand_as(mean)
        else:
            raise ValueError("Unknown noise_std_type")
        self.distribution = Normal(mean, std)

    def evaluate(self, obs: TensorDict) -> torch.Tensor:
        return self.get_teacher_extrinsics(obs)

    def get_teacher_extrinsics(self, obs: TensorDict) -> torch.Tensor:
        priv = torch.cat([obs[k] for k in self.obs_groups["priv"]], dim=-1)
        priv = self.priv_normalizer(priv)

        with torch.no_grad():
            return self.teacher_extrinsics_encoder(priv)

    def update_normalization(self, obs: TensorDict) -> None:

        if isinstance(self.obs_normalizer, EmpiricalNormalization):
            obs_vec = torch.cat(
                [obs[k] for k in self.obs_groups["obs"]],
                dim=-1,
            )
            self.obs_normalizer.update(obs_vec)

        if isinstance(self.priv_normalizer, EmpiricalNormalization):
            priv = torch.cat(
                [obs[k] for k in self.obs_groups["priv"]],
                dim=-1,
            )
            self.priv_normalizer.update(priv)

        if isinstance(self.obs_hist_normalizer, EmpiricalNormalization):
            hist = torch.cat(
                [obs[k] for k in self.obs_groups["obs_hist"]],
                dim=-1,
            )  # (B, T, D)

            B, T, D = hist.shape
            hist_flat = hist.reshape(B, T * D)
            self.obs_hist_normalizer.update(hist_flat)

    def get_hidden_states(self) -> tuple[HiddenState, HiddenState]:
        return None, None

    def detach_hidden_states(self, dones: torch.Tensor | None = None) -> None:
        pass
        
    def load_state_dict(self, state_dict: dict, strict: bool = True) -> bool:
        """
        Returns:
            True  -> resume stage-2
            False -> loaded stage-1 PPO checkpoint
        """

        keys = list(state_dict.keys())

        # CASE 1 — Stage-2 checkpoint (student exists)
        if any(k.startswith("student_extrinsics_encoder.") for k in keys):

            super().load_state_dict(state_dict, strict=strict)
            self.loaded_teacher = True
            return True

        # CASE 2 — Stage-1 PPO checkpoint
        elif any(k.startswith("actor.") for k in keys):

            teacher_sd = {}
            actor_sd = {}
            actor_norm_sd = {}
            priv_norm_sd = {}

            for k, v in state_dict.items():

                # ---------------- Teacher ----------------
                if k.startswith("teacher_extrinsics_encoder."):
                    teacher_sd[k.replace("teacher_extrinsics_encoder.", "")] = v
                    continue

                # ---------------- Actor ----------------
                if k.startswith("actor."):
                    actor_sd[k.replace("actor.", "")] = v
                    continue

                # ---------------- Actor obs normalizer ----------------
                if k.startswith("actor_obs_normalizer."):
                    actor_norm_sd[k.replace("actor_obs_normalizer.", "")] = v
                    continue

                # ---------------- Priv normalizer ----------------
                if k.startswith("priv_obs_normalizer."):
                    priv_norm_sd[k.replace("priv_obs_normalizer.", "")] = v
                    continue

                # ---------------- Noise ----------------
                if k in ("std", "log_std"):
                    setattr(self, k, nn.Parameter(v.clone()))

            # Load weights
            self.teacher_extrinsics_encoder.load_state_dict(teacher_sd, strict=False)
            self.actor.load_state_dict(actor_sd, strict=False)

            # Load normalizers
            if isinstance(self.obs_normalizer, EmpiricalNormalization) and len(actor_norm_sd) > 0:
                self.obs_normalizer.load_state_dict(actor_norm_sd, strict=False)

            if isinstance(self.priv_normalizer, EmpiricalNormalization) and len(priv_norm_sd) > 0:
                self.priv_normalizer.load_state_dict(priv_norm_sd, strict=False)

            self.teacher_extrinsics_encoder.eval()
            self.actor.eval()

            self.loaded_teacher = True
            return False

        # Unknown checkpoint
        else:
            raise ValueError("Unrecognized checkpoint format")

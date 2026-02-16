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

from rsl_rl.networks import MLP, EmpiricalNormalization, HiddenState


class ExtrinsicsStudentTeacher(nn.Module):
    is_recurrent: bool = False

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        num_actions: int,
        extrinsics_output_dims: int,
        extrinsics_hidden_dims: tuple[int] | list[int] = [256, 256, 256],
        base_obs_normalization: bool = False,
        base_policy_hidden_dims: tuple[int] | list[int] = [256, 256, 256],
        activation: str = "elu",
        **kwargs: dict[str, Any],
    ) -> None:
        if kwargs:
            print(
                "ExtrinsicsStudentTeacher.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs])
            )
        super().__init__()
        # --- Sanity checks ---
        required_groups = ["proprio", "priv", "other_obs"]
        for g in required_groups:
            if g not in obs_groups:
                raise ValueError(f"Missing '{g}' in obs_groups.")
            
        self.loaded_teacher = False  # Indicates if teacher has been loaded
        self.obs_groups = obs_groups

        # Proprio (student extrinsics input)
        proprio_dim = 0
        for key in obs_groups["proprio"]:
            assert len(obs[key].shape) == 2, "Only 1D observations supported."
            proprio_dim += obs[key].shape[-1]
        # Privileged (teacher extrinsics input)
        priv_dim = 0
        for key in obs_groups["priv"]:
            assert len(obs[key].shape) == 2, "Only 1D observations supported."
            priv_dim += obs[key].shape[-1]
        # other_obs observations (shared)
        other_dim = 0
        for key in obs_groups["other_obs"]:
            assert len(obs[key].shape) == 2, "Only 1D observations supported."
            other_dim += obs[key].shape[-1]

        # Student extrinsics encoder φ
        self.student_extrinsics_encoder = MLP(
            proprio_dim,
            extrinsics_output_dims,
            extrinsics_hidden_dims,
            activation,
        )
        # Teacher extrinsics encoder μ
        self.teacher_extrinsics_encoder = MLP(
            priv_dim,
            extrinsics_output_dims,
            extrinsics_hidden_dims,
            activation,
        )

        # --- Base policy π input dim ---
        num_base_policy_obs = other_dim + extrinsics_output_dims
        teacher_num_base_policy_obs = other_dim + extrinsics_output_dims

        # Optional dimension consistency check
        if teacher_num_base_policy_obs != num_base_policy_obs:
            raise ValueError(
                f"Base policy input dims mismatch: "
                f"teacher={teacher_num_base_policy_obs}, student={num_base_policy_obs}"
            )

        # base
        self.base_policy = MLP(num_base_policy_obs, num_actions, base_policy_hidden_dims, activation)
        self.base_policy.eval()  # We only train the extrinsics encoder, so we can set the base policy to eval mode to save memory and slightly speed up training
        for p in self.base_policy.parameters():
            p.requires_grad = False
        # base observation normalization
        self.base_obs_normalization = base_obs_normalization
        if base_obs_normalization:
            self.base_obs_normalizer = EmpiricalNormalization(num_base_policy_obs)
        else:
            self.base_obs_normalizer = torch.nn.Identity()

        # Teacher
        self.teacher_extrinsics_encoder.eval()
        for p in self.teacher_extrinsics_encoder.parameters():
            p.requires_grad = False
        print(f"base policy MLP: {self.base_policy}")
        print(f"Student extrinsics encoder: {self.student_extrinsics_encoder}")
        print(f"Teacher extrinsics encoder: {self.teacher_extrinsics_encoder}")
        # Teacher observation normalization

        # Action distribution
        # Note: Populated in update_distribution
        self.distribution = None

        # Disable args validation for speedup
        Normal.set_default_validate_args(False)

    def reset(
        self, dones: torch.Tensor | None = None, hidden_states: tuple[HiddenState, HiddenState] = (None, None)
    ) -> None:
        pass

    def forward(self) -> NoReturn:
        raise NotImplementedError

    def act_inference(self, obs: TensorDict) -> torch.Tensor:
        extrinsics_student_obs = self.forward_student_latent(obs)
        other_obs = self.get_other_obs(obs)
        cat_obs = torch.cat([extrinsics_student_obs, other_obs], dim=-1)
        norm_obs = self.base_obs_normalizer(cat_obs)
        return self.base_policy(norm_obs)

    def forward_student_latent(self, obs):
        proprio_obs = self.get_proprio_obs(obs)
        return self.student_extrinsics_encoder(proprio_obs)

    def forward_teacher_latent(self, obs):
        with torch.no_grad():
            priv_obs = self.get_priv_obs(obs)
            return self.teacher_extrinsics_encoder(priv_obs)

    def get_priv_obs(self, obs: TensorDict) -> torch.Tensor:
        obs_list = [obs[k] for k in self.obs_groups["priv"]]
        return torch.cat(obs_list, dim=-1)

    def get_proprio_obs(self, obs: TensorDict) -> torch.Tensor:
        obs_list = [obs[k] for k in self.obs_groups["proprio"]]
        return torch.cat(obs_list, dim=-1)

    def get_other_obs(self, obs: TensorDict) -> torch.Tensor:
        obs_list = [obs[k] for k in self.obs_groups["other_obs"]]
        return torch.cat(obs_list, dim=-1)

    def get_hidden_states(self) -> tuple[HiddenState, HiddenState]:
        return None, None

    def detach_hidden_states(self, dones: torch.Tensor | None = None) -> None:
        pass

    def train(self, mode: bool = True) -> None:
        super().train(mode)
        # Make sure teacher is in eval mode
        self.teacher_extrinsics_encoder.eval()
        self.base_policy.eval()

    def load_state_dict(self, state_dict: dict, strict: bool = True) -> bool:
        """Load the parameters of the student and teacher networks.

        Args:
            state_dict: State dictionary of the model.
            strict: Whether to strictly enforce that the keys in `state_dict` match the keys returned by this module's
                :meth:`state_dict` function.

        Returns:
            Whether this training resumes a previous training. This flag is used by the :func:`load` function of
                :class:`OnPolicyRunner` to determine how to load further parameters.
        """
        # Check if state_dict contains teacher and student or just teacher parameters
        if any("actor" in key for key in state_dict):  # Load parameters from rl training
            # Rename keys to match teacher and remove critic parameters
            teacher_state_dict = {}
            teacher_obs_normalizer_state_dict = {}
            for key, value in state_dict.items():
                if "actor." in key:
                    teacher_state_dict[key.replace("actor.", "")] = value
                if "actor_obs_normalizer." in key:
                    teacher_obs_normalizer_state_dict[key.replace("actor_obs_normalizer.", "")] = value
            self.teacher.load_state_dict(teacher_state_dict, strict=strict)
            self.teacher_obs_normalizer.load_state_dict(teacher_obs_normalizer_state_dict, strict=strict)
            # Set flag for successfully loading the parameters
            self.loaded_teacher = True
            self.teacher.eval()
            self.teacher_obs_normalizer.eval()
            return False  # Training does not resume
        elif any("student" in key for key in state_dict):  # Load parameters from distillation training
            super().load_state_dict(state_dict, strict=strict)
            # Set flag for successfully loading the parameters
            self.loaded_teacher = True
            self.teacher.eval()
            self.teacher_obs_normalizer.eval()
            return True  # Training resumes
        else:
            raise ValueError("state_dict does not contain student or teacher parameters")



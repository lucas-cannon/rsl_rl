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
from rsl_rl.networks import CNN, MLP, EmpiricalNormalization, HiddenState

class StudentTeacherCNN(nn.Module):
    is_recurrent: bool = False

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        num_actions: int,
        student_obs_normalization: bool = False,
        teacher_obs_normalization: bool = False,
        student_hidden_dims: tuple[int] | list[int] = [256, 256, 256],
        teacher_hidden_dims: tuple[int] | list[int] = [256, 256, 256],
        student_cnn_cfg: dict[str, dict] | dict | None = None,
        activation: str = "elu",
        init_noise_std: float = 0.1,
        noise_std_type: str = "scalar",
        **kwargs: dict[str, Any],
    ) -> None:
        if kwargs:
            print(
                "StudentTeacher.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs])
            )
        super().__init__()

        self.loaded_teacher = False  # Indicates if teacher has been loaded

        # Get the observation dimensions
        self.obs_groups = obs_groups
        # num_student_obs = 0
        num_student_obs_1d = 0
        self.student_obs_groups_1d = []
        student_in_dims_2d = []  # H, W (input image height and width for CNN)
        student_in_channels_2d = []  # C (input image channels for CNN)
        self.student_obs_groups_2d = []

        
        for obs_group in obs_groups["policy"]:
            # assert len(obs[obs_group].shape) == 2, "The StudentTeacher module only supports 1D observations."
            # num_student_obs += obs[obs_group].shape[-1]
            if len(obs[obs_group].shape) == 4:  # B, C, H, W
                self.student_obs_groups_2d.append(obs_group)
                student_in_dims_2d.append(obs[obs_group].shape[2:4])  # H, W
                student_in_channels_2d.append(obs[obs_group].shape[1])  # C
            elif len(obs[obs_group].shape) == 2:  # B, C
                self.student_obs_groups_1d.append(obs_group)
                num_student_obs_1d += obs[obs_group].shape[-1]
            else:
                raise ValueError(f"Invalid observation shape for {obs_group}: {obs[obs_group].shape}")
        
        num_teacher_obs = 0
        # TODO: For future extension to 2D obs for teacher
        # num_teacher_obs_1d = 0
        # self.teacher_obs_groups_1d = []
        # teacher_in_dims_2d = []  # H, W (input image height and
        # teacher_in_channels_2d = [] # C (input image channels for CNN)
        # self.teacher_obs_groups_2d = []
        for obs_group in obs_groups["teacher"]:
            assert len(obs[obs_group].shape) == 2, "The StudentTeacher module only supports 1D observations for the teacher."
            num_teacher_obs += obs[obs_group].shape[-1]

        # Student CNN
        if self.student_obs_groups_2d:
            # Resolve the student CNN configuration
            assert student_cnn_cfg is not None, "A student CNN configuration is required for 2D student observations."
            # If a single configuration dictionary is provided, create a dictionary for each 2D observation group
            if not all(isinstance(v, dict) for v in student_cnn_cfg.values()):
                student_cnn_cfg = {group: student_cnn_cfg for group in self.student_obs_groups_2d}
            # Check that the number of configs matches the number of observation groups
            assert len(student_cnn_cfg) == len(self.student_obs_groups_2d), (
                "The number of CNN configurations must match the number of 2D student observations."
            )

            # Create CNNs for each 2D student observation
            self.student_cnns = nn.ModuleDict()
            encoding_dim = 0
            for idx, obs_group in enumerate(self.student_obs_groups_2d):
                self.student_cnns[obs_group] = CNN(
                    input_dim=student_in_dims_2d[idx],
                    input_channels=student_in_channels_2d[idx],
                    **student_cnn_cfg[obs_group],
                )
                print(f"Student CNN for {obs_group}: {self.student_cnns[obs_group]}")
                # Get the output dimension of the CNN
                if self.student_cnns[obs_group].output_channels is None:
                    encoding_dim += int(self.student_cnns[obs_group].output_dim)  # type: ignore
                else:
                    raise ValueError("The output of the student CNN must be flattened before passing it to the MLP.")
        else:
            self.student_cnns = None
            encoding_dim = 0

        # Student MLP
        self.student = MLP(num_student_obs_1d + encoding_dim, num_actions, student_hidden_dims, activation)
        print(f"Student MLP: {self.student}")


        # Student observation normalization
        self.student_obs_normalization = student_obs_normalization
        if student_obs_normalization:
            self.student_obs_normalizer = EmpiricalNormalization(num_student_obs_1d)
        else:
            self.student_obs_normalizer = torch.nn.Identity()

        # Teacher 1D input
        self.teacher = MLP(num_teacher_obs, num_actions, teacher_hidden_dims, activation)
        self.teacher.eval()
        print(f"Teacher MLP: {self.teacher}")

        # Teacher observation normalization
        self.teacher_obs_normalization = teacher_obs_normalization
        if teacher_obs_normalization:
            self.teacher_obs_normalizer = EmpiricalNormalization(num_teacher_obs)
        else:
            self.teacher_obs_normalizer = torch.nn.Identity()

        # Action noise
        self.noise_std_type = noise_std_type
        if self.noise_std_type == "scalar":
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        elif self.noise_std_type == "log":
            self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")

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

    @property
    def action_mean(self) -> torch.Tensor:
        return self.distribution.mean

    @property
    def action_std(self) -> torch.Tensor:
        return self.distribution.stddev

    @property
    def entropy(self) -> torch.Tensor:
        return self.distribution.entropy().sum(dim=-1)

    def _update_distribution(self, obs: torch.Tensor) -> None:
        # Compute mean
        mean = self.student(obs)
        # Compute standard deviation
        if self.noise_std_type == "scalar":
            std = self.std.expand_as(mean)
        elif self.noise_std_type == "log":
            std = torch.exp(self.log_std).expand_as(mean)
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
        # Create distribution
        self.distribution = Normal(mean, std)

    def act(self, obs: TensorDict) -> torch.Tensor:
        mlp_obs, cnn_obs = self.get_student_obs(obs)
        mlp_obs = self.student_obs_normalizer(mlp_obs)
        if self.student_cnns is not None:
            # Encode the 2D student observations
            cnn_enc_list = [self.student_cnns[obs_group](cnn_obs[obs_group]) for obs_group in self.student_obs_groups_2d]
            cnn_enc = torch.cat(cnn_enc_list, dim=-1)
            # Concatenate to the MLP observations
            mlp_obs = torch.cat([mlp_obs, cnn_enc], dim=-1)

        self._update_distribution(mlp_obs)  # TODO: check if this input (tensor or tensor.dict) is correct
        return self.distribution.sample()

    def act_inference(self, obs: TensorDict) -> torch.Tensor:
        mlp_obs, cnn_obs = self.get_student_obs(obs)
        mlp_obs = self.student_obs_normalizer(mlp_obs)
        if self.student_cnns is not None:
            # Encode the 2D actor observations
            cnn_enc_list = [self.student_cnns[obs_group](cnn_obs[obs_group]) for obs_group in self.student_obs_groups_2d]
            cnn_enc = torch.cat(cnn_enc_list, dim=-1)
            # Concatenate to the MLP observations
            mlp_obs = torch.cat([mlp_obs, cnn_enc], dim=-1)

        return self.student(mlp_obs)

    def evaluate(self, obs: TensorDict) -> torch.Tensor:
        # For getting teacher actions 
        obs = self.get_teacher_obs(obs)
        obs = self.teacher_obs_normalizer(obs)
        with torch.no_grad():
            return self.teacher(obs)

    # def get_student_obs(self, obs: TensorDict) -> torch.Tensor:
    #     obs_list = [obs[obs_group] for obs_group in self.obs_groups["policy"]]
    #     return torch.cat(obs_list, dim=-1)

    def get_student_obs(self, obs: TensorDict) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        obs_list_1d = [obs[obs_group] for obs_group in self.student_obs_groups_1d]
        obs_dict_2d = {}
        for obs_group in self.student_obs_groups_2d:
            obs_dict_2d[obs_group] = obs[obs_group]
        return torch.cat(obs_list_1d, dim=-1), obs_dict_2d
    
    def get_teacher_obs(self, obs: TensorDict) -> torch.Tensor:
        obs_list = [obs[obs_group] for obs_group in self.obs_groups["teacher"]]
        return torch.cat(obs_list, dim=-1)

    def get_hidden_states(self) -> tuple[HiddenState, HiddenState]:
        return None, None

    def detach_hidden_states(self, dones: torch.Tensor | None = None) -> None:
        pass

    def train(self, mode: bool = True) -> None:
        super().train(mode)
        # Make sure teacher is in eval mode
        self.teacher.eval()
        self.teacher_obs_normalizer.eval()

    def update_normalization(self, obs: TensorDict) -> None:
        if self.student_obs_normalization:
            student_1d_obs, _ = self.get_student_obs(obs)
            self.student_obs_normalizer.update(student_1d_obs)

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
        # set_trace()
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

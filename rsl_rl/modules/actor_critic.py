# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from tensordict import TensorDict
from torch.distributions import Normal
from typing import Any, NoReturn

from rsl_rl.networks import MLP, EmpiricalNormalization
from ipdb import set_trace

class ActorCritic(nn.Module):
    is_recurrent: bool = False

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        num_actions: int,
        actor_obs_normalization: bool = False,
        critic_obs_normalization: bool = False,
        actor_hidden_dims: tuple[int] | list[int] = [256, 256, 256],
        critic_hidden_dims: tuple[int] | list[int] = [256, 256, 256],
        activation: str = "elu",
        init_noise_std: float = 1.0,
        noise_std_type: str = "scalar",
        state_dependent_std: bool = False,
        action_distribution: str = "normal",
        **kwargs: dict[str, Any],
    ) -> None:
        if kwargs:
            print(
                "ActorCritic.__init__ got unexpected arguments, which will be ignored: " + str([key for key in kwargs])
            )
        super().__init__()

        # Get the observation dimensions
        self.obs_groups = obs_groups
        num_actor_obs = 0
        for obs_group in obs_groups["policy"]:
            assert len(obs[obs_group].shape) == 2, "The ActorCritic module only supports 1D observations."
            num_actor_obs += obs[obs_group].shape[-1]
        num_critic_obs = 0
        for obs_group in obs_groups["critic"]:
            assert len(obs[obs_group].shape) == 2, "The ActorCritic module only supports 1D observations."
            num_critic_obs += obs[obs_group].shape[-1]

        # Actor
        self.state_dependent_std = state_dependent_std
        if self.state_dependent_std:
            self.actor = MLP(num_actor_obs, [2, num_actions], actor_hidden_dims, activation)
        else:
            self.actor = MLP(num_actor_obs, num_actions, actor_hidden_dims, activation)
        print(f"Actor MLP: {self.actor}")

        # Actor observation normalization
        self.actor_obs_normalization = actor_obs_normalization
        if actor_obs_normalization:
            self.actor_obs_normalizer = EmpiricalNormalization(num_actor_obs)
        else:
            self.actor_obs_normalizer = torch.nn.Identity()

        # Critic
        self.critic = MLP(num_critic_obs, 1, critic_hidden_dims, activation)
        print(f"Critic MLP: {self.critic}")

        # Critic observation normalization
        self.critic_obs_normalization = critic_obs_normalization
        if critic_obs_normalization:
            self.critic_obs_normalizer = EmpiricalNormalization(num_critic_obs)
        else:
            self.critic_obs_normalizer = torch.nn.Identity()

        # Action noise
        if action_distribution not in {"normal", "tanh_normal"}:
            raise ValueError(
                "action_distribution must be 'normal' or 'tanh_normal', got "
                f"{action_distribution!r}."
            )
        self.action_distribution = action_distribution
        self.noise_std_type = noise_std_type
        if self.state_dependent_std:
            torch.nn.init.zeros_(self.actor[-2].weight[num_actions:])
            if self.noise_std_type == "scalar":
                torch.nn.init.constant_(self.actor[-2].bias[num_actions:], init_noise_std)
            elif self.noise_std_type == "log":
                torch.nn.init.constant_(
                    self.actor[-2].bias[num_actions:], torch.log(torch.tensor(init_noise_std + 1e-7))
                )
            else:
                raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
        else:
            if self.noise_std_type == "scalar":
                self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
            elif self.noise_std_type == "log":
                self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
            else:
                raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")

        # Action distribution
        # Note: Populated in update_distribution
        self.distribution = None
        self._entropy = None
        self._latent_action = None
        self._sampled_action = None

        # Disable args validation for speedup
        Normal.set_default_validate_args(False)

    def reset(self, dones: torch.Tensor | None = None) -> None:
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
    def deterministic_action(self) -> torch.Tensor:
        """Bounded action selected by deterministic inference for the current observations."""
        return self._squash_action(self.distribution.mean)

    @property
    def latent_action(self) -> torch.Tensor:
        """Last sampled pre-transform action, for saturation diagnostics."""
        if self._latent_action is None:
            raise RuntimeError("act() must be called before reading the latent action.")
        return self._latent_action

    @property
    def entropy(self) -> torch.Tensor:
        if self.action_distribution == "tanh_normal":
            if self._entropy is None:
                raise RuntimeError("act() must be called before reading squashed-policy entropy.")
            return self._entropy
        return self.distribution.entropy().sum(dim=-1)

    @staticmethod
    def _tanh_log_abs_det_jacobian(latent_action: torch.Tensor) -> torch.Tensor:
        """Stable ``log(1 - tanh(x)^2)`` used by the transformed density."""
        log_two = torch.log(torch.tensor(2.0, device=latent_action.device))
        return 2.0 * (log_two - latent_action - F.softplus(-2.0 * latent_action))

    def _sample_action(self) -> torch.Tensor:
        if self.action_distribution == "normal":
            self._entropy = None
            self._latent_action = self.distribution.sample()
            self._sampled_action = self._latent_action
            return self._sampled_action
        latent_action = self.distribution.rsample()
        self._latent_action = latent_action
        log_det = self._tanh_log_abs_det_jacobian(latent_action)
        # Reparameterized Monte-Carlo estimate of the transformed entropy.
        self._entropy = -(self.distribution.log_prob(latent_action) - log_det).sum(dim=-1)
        self._sampled_action = torch.tanh(latent_action)
        return self._sampled_action

    def _squash_action(self, latent_action: torch.Tensor) -> torch.Tensor:
        if self.action_distribution == "tanh_normal":
            return torch.tanh(latent_action)
        return latent_action

    def _update_distribution(self, obs: torch.Tensor) -> None:
        if self.state_dependent_std:
            # Compute mean and standard deviation
            mean_and_std = self.actor(obs)
            if self.noise_std_type == "scalar":
                mean, std = torch.unbind(mean_and_std, dim=-2)
            elif self.noise_std_type == "log":
                mean, log_std = torch.unbind(mean_and_std, dim=-2)
                std = torch.exp(log_std)
            else:
                raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
        else:
            # Compute mean
            mean = self.actor(obs)
            # Compute standard deviation
            if self.noise_std_type == "scalar":
                std = self.std.expand_as(mean)
            elif self.noise_std_type == "log":
                std = torch.exp(self.log_std).expand_as(mean)
            else:
                raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
        # Create distribution
        self.distribution = Normal(mean, std)

    def act(self, obs: TensorDict, **kwargs: dict[str, Any]) -> torch.Tensor:
        obs = self.get_actor_obs(obs)
        obs = self.actor_obs_normalizer(obs)
        self._update_distribution(obs)
        return self._sample_action()

    def act_inference(self, obs: TensorDict) -> torch.Tensor:
        obs = self.get_actor_obs(obs)
        obs = self.actor_obs_normalizer(obs)
        # set_trace()
        if self.state_dependent_std:
            latent_action = self.actor(obs)[..., 0, :]
        else:
            latent_action = self.actor(obs)
        return self._squash_action(latent_action)

    def evaluate(self, obs: TensorDict, **kwargs: dict[str, Any]) -> torch.Tensor:
        obs = self.get_critic_obs(obs)
        obs = self.critic_obs_normalizer(obs)
        return self.critic(obs)

    def get_actor_obs(self, obs: TensorDict) -> torch.Tensor:
        obs_list = [obs[obs_group] for obs_group in self.obs_groups["policy"]]
        return torch.cat(obs_list, dim=-1)

    def get_critic_obs(self, obs: TensorDict) -> torch.Tensor:
        obs_list = [obs[obs_group] for obs_group in self.obs_groups["critic"]]
        return torch.cat(obs_list, dim=-1)

    def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        if self.action_distribution == "normal":
            return self.distribution.log_prob(actions).sum(dim=-1)
        if actions is self._sampled_action:
            log_det = self._tanh_log_abs_det_jacobian(self._latent_action)
            return (self.distribution.log_prob(self._latent_action) - log_det).sum(dim=-1)
        # atanh is undefined at +/-1. Saturated float values are mapped to the
        # nearest representable interior value for a finite, consistent density.
        eps = torch.finfo(actions.dtype).eps
        bounded_actions = actions.clamp(min=-1.0 + eps, max=1.0 - eps)
        latent_actions = torch.atanh(bounded_actions)
        log_det = self._tanh_log_abs_det_jacobian(latent_actions)
        return (self.distribution.log_prob(latent_actions) - log_det).sum(dim=-1)

    def update_normalization(self, obs: TensorDict) -> None:
        if self.actor_obs_normalization:
            actor_obs = self.get_actor_obs(obs)
            self.actor_obs_normalizer.update(actor_obs)
        if self.critic_obs_normalization:
            critic_obs = self.get_critic_obs(obs)
            self.critic_obs_normalizer.update(critic_obs)

    def load_state_dict(self, state_dict: dict, strict: bool = True) -> bool:
        """
        Returns:
            True  -> resume PPO training (actor + critic present)
            False -> student-only checkpoint (actor loaded, critic missing)
        """
        keys = list(state_dict.keys())
        #######################################################################
        # CASE 1 — Distillation student checkpoint (student.*, student_cnns.*)
        #######################################################################
        if any(k.startswith("student.") for k in keys) or any(k.startswith("student_cnns.") for k in keys):
            print("[ActorCritic] Distillation checkpoint detected → loading student into actor only")

            new_sd = {}

            for k, v in state_dict.items():
                # CNN remapping
                if k.startswith("student_cnns."):
                    new_sd[k.replace("student_cnns.", "actor_cnns.")] = v
                    continue

                # MLP remapping
                if k.startswith("student."):
                    new_sd[k.replace("student.", "actor.")] = v
                    continue

                # Noise parameters
                if k in ("std", "log_std"):
                    new_sd[k] = v

            # Load only the actor parameters
            super().load_state_dict(new_sd, strict=False)

            print("[ActorCritic] Loaded student policy → critic intentionally NOT loaded.")
            return False      # <<< IMPORTANT: tells PPO NOT to load optimizer

        #######################################################################
        # CASE 2 — PPO checkpoint (actor.*, critic.*)
        #######################################################################
        if any(k.startswith("actor.") for k in keys) and any(k.startswith("critic.") for k in keys):
            print("[ActorCritic] PPO checkpoint detected → loading actor + critic")
            super().load_state_dict(state_dict, strict=strict)
            return True       # resume PPO normally

        #######################################################################
        # CASE 3 — Actor-only PPO checkpoint (rare but possible)
        #######################################################################
        if any(k.startswith("actor.") for k in keys) and not any(k.startswith("critic.") for k in keys):
            print("[ActorCritic] WARNING: actor-only checkpoint loaded → critic reset")
            super().load_state_dict(state_dict, strict=False)
            return False      # new critic → new optimizer

        #######################################################################
        # CASE 4 — Unrecognized checkpoint
        #######################################################################
        raise RuntimeError("Unrecognized checkpoint format. Keys: " + str(keys[:20]))

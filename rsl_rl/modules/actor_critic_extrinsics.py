from __future__ import annotations

import torch
from tensordict import TensorDict

from rsl_rl.networks import MLP, EmpiricalNormalization
from torch import nn
from torch.distributions import Normal
from .actor_critic import ActorCritic
from ipdb import set_trace

class ActorCriticExtrinsics(ActorCritic):
    is_recurrent: bool = False

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        num_actions: int,
        extrinsics_output_dims: int,
        extrinsics_hidden_dims=(256, 128),
        actor_obs_normalization: bool = False,
        priv_obs_normalization: bool = False,
        critic_obs_normalization: bool = False,
        actor_hidden_dims=(512, 256, 128),
        critic_hidden_dims=(512, 256, 128),
        activation="elu",
        last_activation=None,
        init_noise_std=1.0,
        noise_std_type="scalar",
        state_dependent_std=False,
        **kwargs,
    ):
        super(ActorCritic, self).__init__()

        # --- Required groups ---
        required = ["priv", "obs", "critic"]
        for g in required:
            if g not in obs_groups:
                raise ValueError(f"Missing '{g}' in obs_groups.")
        # ---- Enforce 1D observations only ----
        for group_name in ["priv", "obs", "critic"]:
            for key in obs_groups[group_name]:
                if len(obs[key].shape) != 2:
                    raise ValueError(
                        f"ActorCriticExtrinsics only supports 1D observations. "
                        f"Observation '{key}' has shape {obs[key].shape}."
                    )
        self.obs_groups = obs_groups
        self.state_dependent_std = state_dependent_std
        self.noise_std_type = noise_std_type

        # ---- Compute dimensions ----
        priv_dim = sum(obs[k].shape[-1] for k in obs_groups["priv"])
        obs_dim = sum(obs[k].shape[-1] for k in obs_groups["obs"])
        critic_dim = sum(obs[k].shape[-1] for k in obs_groups["critic"])

        # ---- Build extrinsics encoder μ ----
        self.teacher_extrinsics_encoder = MLP(
            priv_dim,
            extrinsics_output_dims,
            extrinsics_hidden_dims,
            activation,
            last_activation
        )
        print(f"Teacher extrinsics encoder: {self.teacher_extrinsics_encoder}")

        # ---- Actor (obs + z) ----
        actor_input_dim = obs_dim + extrinsics_output_dims

        if self.state_dependent_std:
            self.actor = MLP(actor_input_dim, [2, num_actions], actor_hidden_dims, activation)
        else:
            self.actor = MLP(actor_input_dim, num_actions, actor_hidden_dims, activation)

        print(f"Actor MLP: {self.actor}")

        # ---- Actor normalization ----
        self.actor_obs_normalization = actor_obs_normalization
        self.priv_obs_normalization = priv_obs_normalization

        if actor_obs_normalization:
            self.actor_obs_normalizer = EmpiricalNormalization(obs_dim)
        else:
            self.actor_obs_normalizer = torch.nn.Identity()
        if priv_obs_normalization:
            self.actor_priv_normalizer = EmpiricalNormalization(priv_dim)
        else:
            self.actor_priv_normalizer = torch.nn.Identity()
        # ---- Critic (unchanged design) ----
        self.critic = MLP(critic_dim, 1, critic_hidden_dims, activation)
        print(f"Critic MLP: {self.critic}")

        self.critic_obs_normalization = critic_obs_normalization
        if critic_obs_normalization:
            self.critic_obs_normalizer = EmpiricalNormalization(critic_dim)
        else:
            self.critic_obs_normalizer = torch.nn.Identity()

        # ---- Action noise ----
        if not self.state_dependent_std:
            if self.noise_std_type == "scalar":
                self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
            elif self.noise_std_type == "log":
                self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
            else:
                raise ValueError("Unknown noise_std_type")

        self.distribution = None
        Normal.set_default_validate_args(False)

    def get_actor_obs(self, obs: TensorDict) -> tuple[torch.Tensor, torch.Tensor]:
        priv = torch.cat([obs[k] for k in self.obs_groups["priv"]], dim=-1)
        priv = self.actor_priv_normalizer(priv)
        extrinsics_vec = self.teacher_extrinsics_encoder(priv)
        obs = torch.cat([obs[k] for k in self.obs_groups["obs"]], dim=-1)
        return obs, extrinsics_vec

    def act(self, obs: TensorDict, **kwargs: dict[str, Any]) -> torch.Tensor:
        obs, extrinsics_vec = self.get_actor_obs(obs)
        obs = self.actor_obs_normalizer(obs)
        self._update_distribution(obs, extrinsics_vec)
        return self.distribution.sample()

    def act_inference(self, obs: TensorDict) -> torch.Tensor:
        obs, extrinsics_vec = self.get_actor_obs(obs)
        obs = self.actor_obs_normalizer(obs)
        mlp_obs = torch.cat([obs, extrinsics_vec], dim=-1)
        if self.state_dependent_std:
            return self.actor(mlp_obs)[..., 0, :]
        else:
            return self.actor(mlp_obs)

    def _update_distribution(self, obs: torch.Tensor, extrinsics_vec: torch.Tensor) -> None:
        mlp_obs = torch.cat([obs, extrinsics_vec], dim=-1)
        super()._update_distribution(mlp_obs)

    def update_normalization(self, obs: TensorDict) -> None:
        if self.actor_obs_normalization:
            obs_vec = torch.cat([obs[k] for k in self.obs_groups["obs"]], dim=-1)
            self.actor_obs_normalizer.update(obs_vec)
        if self.priv_obs_normalization:
            priv = torch.cat([obs[k] for k in self.obs_groups["priv"]], dim=-1)
            self.actor_priv_normalizer.update(priv)
        if self.critic_obs_normalization:
            critic_obs = self.get_critic_obs(obs)
            self.critic_obs_normalizer.update(critic_obs)

    # -------------------------------------------------
    # Recurrent compatibility (non-recurrent policy)
    # -------------------------------------------------

    def get_hidden_states(self):
        # Non-recurrent model → no hidden states
        return None, None


    def detach_hidden_states(self, dones: torch.Tensor | None = None):
        # Nothing to detach
        pass


    def reset(self, dones: torch.Tensor | None = None):
        # Nothing to reset
        pass
# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import os
import statistics
import time
import torch
import warnings
from collections import deque
from collections.abc import Callable
from typing import Any
from tensordict import TensorDict

import rsl_rl
from rsl_rl.algorithms import PPO
from rsl_rl.env import VecEnv
from rsl_rl.modules import (
    ActorCritic,
    ActorCriticCNN,
    ActorCriticRecurrent,
    ActorCriticExtrinsics,
    resolve_rnd_config,
    resolve_symmetry_config,
)
from rsl_rl.utils import resolve_obs_groups, store_code_state

class OnPolicyRunner:
    """On-policy runner for training and evaluation of actor-critic methods."""

    def __init__(self, env: VecEnv, train_cfg: dict, log_dir: str | None = None, device: str = "cpu") -> None:
        self.cfg = train_cfg
        self.alg_cfg = train_cfg["algorithm"]
        self.policy_cfg = train_cfg["policy"]
        self.device = device
        self.env = env

        # Check if multi-GPU is enabled
        self._configure_multi_gpu()

        # Store training configuration
        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]

        # Query observations from environment for algorithm construction
        obs = self.env.get_observations()
        default_sets = ["critic"]
        if "rnd_cfg" in self.alg_cfg and self.alg_cfg["rnd_cfg"] is not None:
            default_sets.append("rnd_state")
        
        self.cfg["obs_groups"] = resolve_obs_groups(obs, self.cfg["obs_groups"], default_sets)

        # Create the algorithm
        self.alg = self._construct_algorithm(obs)

        # Decide whether to disable logging
        # Note: We only log from the process with rank 0 (main process)
        self.disable_logs = self.is_distributed and self.gpu_global_rank != 0

        # Logging
        self.log_dir = log_dir
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0
        self.last_iteration_metrics: dict[str, Any] | None = None
        self.git_status_repos = [rsl_rl.__file__]

    def learn(
        self,
        num_learning_iterations: int,
        init_at_random_ep_len: bool = False,
        iteration_callback: Callable[[dict[str, Any]], bool | None] | None = None,
    ) -> None:
        """Train for at most ``num_learning_iterations`` iterations.

        ``iteration_callback`` is invoked after each completed/logged iteration
        with the runner's rolling episode statistics and policy diagnostics.  A
        truthy return requests a clean early stop.  Keeping the callback inside
        one ``learn`` call is important: the rolling 100-episode buffers and the
        best-policy tracker remain continuous for the whole run.

        The argument is optional and therefore backwards compatible with the
        ordinary fixed-length RSL-RL training path.
        """
        # Initialize writer
        self._prepare_logging_writer()

        # Randomize initial episode lengths (for exploration)
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )

        # Start learning
        obs = self.env.get_observations().to(self.device)
        self.train_mode()  # switch to train mode (for dropout for example)

        # Book keeping
        ep_infos = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        # Per-completed-episode success flag (1.0 if the episode TERMINATED — reached the goal —
        # rather than timing out). The env terminates only on reached_goal, so success =
        # done & not time_out. Logged as Train/success_rate (mean over the last 100 episodes).
        successbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        # --- Best policy tracking ---
        best_mean_reward = -float("inf")
        best_model_path = os.path.join(self.log_dir, "best_model.pt") if self.log_dir else None

        # Create buffers for logging extrinsic and intrinsic rewards
        if self.alg.rnd:
            erewbuffer = deque(maxlen=100)
            irewbuffer = deque(maxlen=100)
            cur_ereward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
            cur_ireward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        # Ensure all parameters are in-synced
        if self.is_distributed:
            print(f"Synchronizing parameters for rank {self.gpu_global_rank}...")
            self.alg.broadcast_parameters()

        # Start training
        start_iter = self.current_learning_iteration
        tot_iter = start_iter + num_learning_iterations
        for it in range(start_iter, tot_iter):
            completed_episodes_this_iteration = 0
            action_near_bound_count = torch.zeros((), device=self.device)
            latent_action_out_of_bounds_count = torch.zeros((), device=self.device)
            action_out_of_bounds_count = torch.zeros((), device=self.device)
            deterministic_action_near_bound_count = torch.zeros((), device=self.device)
            stochastic_deterministic_squared_error = torch.zeros((), device=self.device)
            action_element_count = 0
            start = time.time()
            # Rollout
            with torch.inference_mode():
                for _ in range(self.num_steps_per_env):
                    # Sample actions
                    actions = self.alg.act(obs)
                    deterministic_actions = getattr(
                        self.alg.policy,
                        "deterministic_action",
                        getattr(self.alg.policy, "action_mean", actions),
                    )
                    latent_actions = getattr(self.alg.policy, "latent_action", actions)
                    action_near_bound_count += (actions.abs() >= 0.95).sum()
                    latent_action_out_of_bounds_count += (latent_actions.abs() > 1.0).sum()
                    action_out_of_bounds_count += (actions.abs() > 1.0).sum()
                    deterministic_action_near_bound_count += (
                        deterministic_actions.abs() >= 0.95
                    ).sum()
                    stochastic_deterministic_squared_error += (
                        actions - deterministic_actions
                    ).square().sum()
                    action_element_count += actions.numel()
                    # Step the environment
                    obs, rewards, dones, extras = self.env.step(actions.to(self.env.device))
                    # Move to device
                    obs, rewards, dones = (obs.to(self.device), rewards.to(self.device), dones.to(self.device))
                    # Process the step
                    self.alg.process_env_step(obs, rewards, dones, extras)
                    # Extract intrinsic rewards (only for logging)
                    intrinsic_rewards = self.alg.intrinsic_rewards if self.alg.rnd else None
                    # Book keeping
                    if self.log_dir is not None:
                        if "episode" in extras:
                            ep_infos.append(extras["episode"])
                        elif "log" in extras:
                            ep_infos.append(extras["log"])
                        # Update rewards
                        if self.alg.rnd:
                            cur_ereward_sum += rewards
                            cur_ireward_sum += intrinsic_rewards
                            cur_reward_sum += rewards + intrinsic_rewards
                        else:
                            cur_reward_sum += rewards
                        # Update episode length
                        cur_episode_length += 1
                        # Clear data for completed episodes
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        completed_episodes_this_iteration += int(new_ids.shape[0])
                        rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                        # Success = ended by TERMINATION (goal), not time-out. dones marks every
                        # ending; extras["time_outs"] marks the truncations, so their complement
                        # among dones is the goal-reached successes (the env's only termination).
                        if "time_outs" in extras:
                            time_outs = extras["time_outs"].to(self.device)
                            success = (dones > 0) & (time_outs <= 0)
                        else:
                            success = torch.zeros_like(dones, dtype=torch.bool)
                        successbuffer.extend(success[new_ids][:, 0].float().cpu().numpy().tolist())
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0
                        if self.alg.rnd:
                            erewbuffer.extend(cur_ereward_sum[new_ids][:, 0].cpu().numpy().tolist())
                            irewbuffer.extend(cur_ireward_sum[new_ids][:, 0].cpu().numpy().tolist())
                            cur_ereward_sum[new_ids] = 0
                            cur_ireward_sum[new_ids] = 0

                stop = time.time()
                collection_time = stop - start
                start = stop

                # Compute returns
                self.alg.compute_returns(obs)

            # Update policy
            loss_dict = self.alg.update()

            stop = time.time()
            learn_time = stop - start
            self.current_learning_iteration = it

            if self.log_dir is not None and not self.disable_logs:
                # Log information
                self.log(locals())
                # -------- Save best model --------
                if rewbuffer:
                    mean_rew = statistics.mean(rewbuffer)

                    if mean_rew > best_mean_reward:
                        best_mean_reward = mean_rew
                        print(
                            f"\033[92m[Best Model] Iter {it}: mean reward improved to {mean_rew:.3f}, saving model.\033[0m"
                        )
                        self.save(best_model_path)
                        # -------- Log best model info to text file --------
                        best_log_path = os.path.join(self.log_dir, "best_policy.txt")
                        with open(best_log_path, "a") as f:
                            f.write(f"Iter {it}: mean_reward = {mean_rew:.6f}\n")

                # Save model
                if it % self.save_interval == 0:
                    self.save(os.path.join(self.log_dir, f"model_{it}.pt"))

            mean_reward = statistics.mean(rewbuffer) if rewbuffer else None
            mean_episode_length = statistics.mean(lenbuffer) if lenbuffer else None
            success_rate = statistics.mean(successbuffer) if successbuffer else None
            mean_noise_std = float(self.alg.policy.action_std.mean().detach().cpu())
            policy_entropy = float(self.alg.policy.entropy.mean().detach().cpu())
            policy_entropy_per_action = policy_entropy / actions.shape[-1]
            action_near_bound_fraction = float(
                (action_near_bound_count / action_element_count).detach().cpu()
            )
            action_out_of_bounds_fraction = float(
                (action_out_of_bounds_count / action_element_count).detach().cpu()
            )
            latent_action_out_of_bounds_fraction = float(
                (latent_action_out_of_bounds_count / action_element_count).detach().cpu()
            )
            deterministic_action_near_bound_fraction = float(
                (deterministic_action_near_bound_count / action_element_count).detach().cpu()
            )
            stochastic_deterministic_action_rmse = float(
                torch.sqrt(stochastic_deterministic_squared_error / action_element_count)
                .detach()
                .cpu()
            )
            total_time = float(getattr(self, "tot_time", 0.0))
            total_timesteps = int(getattr(self, "tot_timesteps", 0))
            samples_per_second = (
                total_timesteps / total_time if total_time > 0.0 else 0.0
            )
            callback_metrics = {
                "iteration": it,
                "mean_reward": mean_reward,
                "best_mean_reward": (
                    None if best_mean_reward == -float("inf") else best_mean_reward
                ),
                "mean_episode_length": mean_episode_length,
                "success_rate": success_rate,
                "completed_episodes": completed_episodes_this_iteration,
                "policy_mean_noise_std": mean_noise_std,
                "policy_entropy": policy_entropy,
                "policy_entropy_per_action": policy_entropy_per_action,
                "action_near_bound_fraction": action_near_bound_fraction,
                "action_out_of_bounds_fraction": action_out_of_bounds_fraction,
                "latent_action_out_of_bounds_fraction": latent_action_out_of_bounds_fraction,
                "deterministic_action_near_bound_fraction": deterministic_action_near_bound_fraction,
                "stochastic_deterministic_action_rmse": stochastic_deterministic_action_rmse,
                "collection_time": collection_time,
                "learning_time": learn_time,
                "iteration_time": collection_time + learn_time,
                "environment_steps": total_timesteps,
                "samples_per_second": samples_per_second,
                # Backward-compatible name used by RSL-RL and existing W&B runs.
                "total_fps": samples_per_second,
            }
            self.last_iteration_metrics = callback_metrics
            stop_requested = bool(
                iteration_callback(callback_metrics)
                if iteration_callback is not None
                else False
            )

            # Clear episode infos
            ep_infos.clear()
            # Save code state
            if it == start_iter and not self.disable_logs:
                # Obtain all the diff files
                git_file_paths = store_code_state(self.log_dir, self.git_status_repos)
                # If possible store them to wandb or neptune
                if self.logger_type in ["wandb", "neptune"] and git_file_paths:
                    for path in git_file_paths:
                        self.writer.save_file(path)

            if stop_requested:
                break

        # Save the final model after training
        if self.log_dir is not None and not self.disable_logs:
            self.save(os.path.join(self.log_dir, f"model_{self.current_learning_iteration}.pt"))

    def log(self, locs: dict, width: int = 80, pad: int = 35) -> None:
        # Compute the collection size
        collection_size = self.num_steps_per_env * self.env.num_envs * self.gpu_world_size
        # Update total time-steps and time
        self.tot_timesteps += collection_size
        self.tot_time += locs["collection_time"] + locs["learn_time"]
        iteration_time = locs["collection_time"] + locs["learn_time"]

        # Log episode information
        if locs["ep_infos"]:
            for key in locs["ep_infos"][0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs["ep_infos"]:
                    # Handle scalar and zero dimensional tensor infos
                    if key not in ep_info:
                        continue
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                value = torch.mean(infotensor)
                # Log to logger and terminal
                if "/" in key:
                    self.writer.add_scalar(key, value, locs["it"])
                else:
                    self.writer.add_scalar("Episode/" + key, value, locs["it"])

        mean_std = self.alg.policy.action_std.mean()
        fps = int(collection_size / (locs["collection_time"] + locs["learn_time"]))

        # Log losses
        for key, value in locs["loss_dict"].items():
            self.writer.add_scalar(f"Loss/{key}", value, locs["it"])
        self.writer.add_scalar("Loss/learning_rate", self.alg.learning_rate, locs["it"])

        # Log noise std
        self.writer.add_scalar("Policy/mean_noise_std", mean_std.item(), locs["it"])
        action_count = locs["action_element_count"]
        self.writer.add_scalar(
            "Policy/action_near_bound_fraction",
            (locs["action_near_bound_count"] / action_count).item(),
            locs["it"],
        )
        self.writer.add_scalar(
            "Policy/action_out_of_bounds_fraction",
            (locs["action_out_of_bounds_count"] / action_count).item(),
            locs["it"],
        )
        self.writer.add_scalar(
            "Policy/latent_action_out_of_bounds_fraction",
            (locs["latent_action_out_of_bounds_count"] / action_count).item(),
            locs["it"],
        )
        self.writer.add_scalar(
            "Policy/deterministic_action_near_bound_fraction",
            (locs["deterministic_action_near_bound_count"] / action_count).item(),
            locs["it"],
        )
        self.writer.add_scalar(
            "Policy/stochastic_deterministic_action_rmse",
            torch.sqrt(locs["stochastic_deterministic_squared_error"] / action_count).item(),
            locs["it"],
        )

        # Log performance
        self.writer.add_scalar("Perf/total_fps", fps, locs["it"])
        self.writer.add_scalar("Perf/collection time", locs["collection_time"], locs["it"])
        self.writer.add_scalar("Perf/learning_time", locs["learn_time"], locs["it"])

        # Log training
        if len(locs["rewbuffer"]) > 0:
            # Separate logging for intrinsic and extrinsic rewards
            if hasattr(self.alg, "rnd") and self.alg.rnd:
                self.writer.add_scalar("Rnd/mean_extrinsic_reward", statistics.mean(locs["erewbuffer"]), locs["it"])
                self.writer.add_scalar("Rnd/mean_intrinsic_reward", statistics.mean(locs["irewbuffer"]), locs["it"])
                self.writer.add_scalar("Rnd/weight", self.alg.rnd.weight, locs["it"])
            # Everything else
            self.writer.add_scalar("Train/mean_reward", statistics.mean(locs["rewbuffer"]), locs["it"])
            self.writer.add_scalar("Train/mean_episode_length", statistics.mean(locs["lenbuffer"]), locs["it"])
            # Fraction of the last 100 completed episodes that reached the goal (vs timed out).
            if len(locs["successbuffer"]) > 0:
                self.writer.add_scalar("Train/success_rate", statistics.mean(locs["successbuffer"]), locs["it"])
            # Episode length in seconds (= mean_episode_length * env step_dt). Computed here
            # over rsl_rl's full lenbuffer (last 100 episodes) so it's a stable curve; an
            # env-side per-reset metric instead sees only the env-group resetting on the last
            # rollout step (~1-2 episodes), which swings wildly around the true mean.
            step_dt = getattr(self.env.unwrapped, "step_dt", None)
            if step_dt is not None:
                self.writer.add_scalar(
                    "Train/mean_episode_time", statistics.mean(locs["lenbuffer"]) * step_dt, locs["it"]
                )
            if self.logger_type != "wandb":  # wandb does not support non-integer x-axis logging
                self.writer.add_scalar("Train/mean_reward/time", statistics.mean(locs["rewbuffer"]), self.tot_time)
                self.writer.add_scalar(
                    "Train/mean_episode_length/time", statistics.mean(locs["lenbuffer"]), self.tot_time
                )

        str = f" \033[1m Learning iteration {locs['it']}/{locs['tot_iter']} \033[0m "

        if len(locs["rewbuffer"]) > 0:
            log_string = (
                f"""{"#" * width}\n"""
                f"""{str.center(width, " ")}\n\n"""
                f"""{"Computation:":>{pad}} {fps:.0f} steps/s (collection: {locs["collection_time"]:.3f}s, learning {
                    locs["learn_time"]:.3f}s)\n"""
            )
        else:
            log_string = (
                f"""{"#" * width}\n"""
                f"""{str.center(width, " ")}\n\n"""
                f"""{"Computation:":>{pad}} {fps:.0f} steps/s (collection: {locs["collection_time"]:.3f}s, learning {
                    locs["learn_time"]:.3f}s)\n"""
            )

        log_string += (
            f"""{"-" * width}\n"""
            f"""{"Total timesteps:":>{pad}} {self.tot_timesteps}\n"""
            f"""{"Iteration time:":>{pad}} {iteration_time:.2f}s\n"""
            f"""{"Time elapsed:":>{pad}} {time.strftime("%H:%M:%S", time.gmtime(self.tot_time))}\n"""
            f"""{"ETA:":>{pad}} {
                time.strftime(
                    "%H:%M:%S",
                    time.gmtime(
                        self.tot_time
                        / (locs["it"] - locs["start_iter"] + 1)
                        * (locs["start_iter"] + locs["num_learning_iterations"] - locs["it"])
                    ),
                )
            }\n"""
        )
        print(log_string)

    def save(self, path: str, infos: dict | None = None) -> None:
        # Save model
        saved_dict = {
            "model_state_dict": self.alg.policy.state_dict(),
            "optimizer_state_dict": self.alg.optimizer.state_dict(),
            "iter": self.current_learning_iteration,
            "infos": infos,
        }
        # Save RND model if used
        if hasattr(self.alg, "rnd") and self.alg.rnd:
            saved_dict["rnd_state_dict"] = self.alg.rnd.state_dict()
            saved_dict["rnd_optimizer_state_dict"] = self.alg.rnd_optimizer.state_dict()
        torch.save(saved_dict, path)

        # Upload model to external logging service
        if self.logger_type in ["neptune", "wandb"] and not self.disable_logs:
            self.writer.save_model(path, self.current_learning_iteration)

    def load(self, path: str, load_optimizer: bool = True, map_location: str | None = None) -> dict:
        loaded_dict = torch.load(path, weights_only=False, map_location=map_location)
        # Load model
        resumed_training = self.alg.policy.load_state_dict(loaded_dict["model_state_dict"])
        # Load RND model if used
        if hasattr(self.alg, "rnd") and self.alg.rnd:
            self.alg.rnd.load_state_dict(loaded_dict["rnd_state_dict"])
        # Load optimizer if used
        if load_optimizer and resumed_training:
            # Algorithm optimizer
            self.alg.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])
            # RND optimizer if used
            if hasattr(self.alg, "rnd") and self.alg.rnd:
                self.alg.rnd_optimizer.load_state_dict(loaded_dict["rnd_optimizer_state_dict"])
        # Load current learning iteration
        if resumed_training:
            self.current_learning_iteration = loaded_dict["iter"]
        return loaded_dict["infos"]

    def get_inference_policy(self, device: str | None = None) -> callable:
        self.eval_mode()  # Switch to evaluation mode (e.g. for dropout)
        if device is not None:
            self.alg.policy.to(device)
        return self.alg.policy.act_inference

    def train_mode(self) -> None:
        # PPO
        self.alg.policy.train()
        # RND
        if hasattr(self.alg, "rnd") and self.alg.rnd:
            self.alg.rnd.train()

    def eval_mode(self) -> None:
        # PPO
        self.alg.policy.eval()
        # RND
        if hasattr(self.alg, "rnd") and self.alg.rnd:
            self.alg.rnd.eval()

    def add_git_repo_to_log(self, repo_file_path: str) -> None:
        self.git_status_repos.append(repo_file_path)

    def _configure_multi_gpu(self) -> None:
        """Configure multi-gpu training."""
        # Check if distributed training is enabled
        self.gpu_world_size = int(os.getenv("WORLD_SIZE", "1"))
        self.is_distributed = self.gpu_world_size > 1

        # If not distributed training, set local and global rank to 0 and return
        if not self.is_distributed:
            self.gpu_local_rank = 0
            self.gpu_global_rank = 0
            self.multi_gpu_cfg = None
            return

        # Get rank and world size
        self.gpu_local_rank = int(os.getenv("LOCAL_RANK", "0"))
        self.gpu_global_rank = int(os.getenv("RANK", "0"))

        # Make a configuration dictionary
        self.multi_gpu_cfg = {
            "global_rank": self.gpu_global_rank,  # Rank of the main process
            "local_rank": self.gpu_local_rank,  # Rank of the current process
            "world_size": self.gpu_world_size,  # Total number of processes
        }

        # Check if user has device specified for local rank
        if self.device != f"cuda:{self.gpu_local_rank}":
            raise ValueError(
                f"Device '{self.device}' does not match expected device for local rank '{self.gpu_local_rank}'."
            )
        # Validate multi-gpu configuration
        if self.gpu_local_rank >= self.gpu_world_size:
            raise ValueError(
                f"Local rank '{self.gpu_local_rank}' is greater than or equal to world size '{self.gpu_world_size}'."
            )
        if self.gpu_global_rank >= self.gpu_world_size:
            raise ValueError(
                f"Global rank '{self.gpu_global_rank}' is greater than or equal to world size '{self.gpu_world_size}'."
            )

        # Initialize torch distributed
        torch.distributed.init_process_group(backend="nccl", rank=self.gpu_global_rank, world_size=self.gpu_world_size)
        # Set device to the local rank
        torch.cuda.set_device(self.gpu_local_rank)

    def _construct_algorithm(self, obs: TensorDict) -> PPO:
        """Construct the actor-critic algorithm."""
        # Resolve RND config
        self.alg_cfg = resolve_rnd_config(self.alg_cfg, obs, self.cfg["obs_groups"], self.env)

        # Resolve symmetry config
        self.alg_cfg = resolve_symmetry_config(self.alg_cfg, self.env)

        # Resolve deprecated normalization config
        if self.cfg.get("empirical_normalization") is not None:
            warnings.warn(
                "The `empirical_normalization` parameter is deprecated. Please set `actor_obs_normalization` and "
                "`critic_obs_normalization` as part of the `policy` configuration instead.",
                DeprecationWarning,
            )
            if self.policy_cfg.get("actor_obs_normalization") is None:
                self.policy_cfg["actor_obs_normalization"] = self.cfg["empirical_normalization"]
            if self.policy_cfg.get("critic_obs_normalization") is None:
                self.policy_cfg["critic_obs_normalization"] = self.cfg["empirical_normalization"]

        # Initialize the policy
        actor_critic_class = eval(self.policy_cfg.pop("class_name"))
        actor_critic: ActorCritic | ActorCriticRecurrent | ActorCriticCNN | ActorCriticExtrinsics = actor_critic_class(
            obs, self.cfg["obs_groups"], self.env.num_actions, **self.policy_cfg
        ).to(self.device)

        # Initialize the algorithm
        alg_class = eval(self.alg_cfg.pop("class_name"))
        alg: PPO = alg_class(actor_critic, device=self.device, **self.alg_cfg, multi_gpu_cfg=self.multi_gpu_cfg)

        # Initialize the storage
        alg.init_storage(
            "rl",
            self.env.num_envs,
            self.num_steps_per_env,
            obs,
            [self.env.num_actions],
        )

        return alg

    def _prepare_logging_writer(self) -> None:
        """Prepare the logging writers."""
        if self.log_dir is not None and self.writer is None and not self.disable_logs:
            # Launch either Tensorboard or Neptune or Tensorboard summary writer, default: Tensorboard.
            self.logger_type = self.cfg.get("logger", "tensorboard")
            self.logger_type = self.logger_type.lower()

            if self.logger_type == "neptune":
                from rsl_rl.utils.neptune_utils import NeptuneSummaryWriter

                self.writer = NeptuneSummaryWriter(log_dir=self.log_dir, flush_secs=10, cfg=self.cfg)
                self.writer.log_config(self.env.cfg, self.cfg, self.alg_cfg, self.policy_cfg)
            elif self.logger_type == "wandb":
                from rsl_rl.utils.wandb_utils import WandbSummaryWriter

                self.writer = WandbSummaryWriter(log_dir=self.log_dir, flush_secs=10, cfg=self.cfg)
                self.writer.log_config(self.env.cfg, self.cfg, self.alg_cfg, self.policy_cfg)
            elif self.logger_type == "tensorboard":
                from torch.utils.tensorboard import SummaryWriter

                self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
            else:
                raise ValueError("Logger type not found. Please choose 'neptune', 'wandb' or 'tensorboard'.")

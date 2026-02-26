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
from tensordict import TensorDict
import json
import pathlib as Path
import pickle
import shutil

import rsl_rl
from rsl_rl.algorithms import PPO
from rsl_rl.env import VecEnv
from rsl_rl.modules import (
    ActorCritic,
    ActorCriticCNN,
    ActorCriticRecurrent,
    resolve_rnd_config,
    resolve_symmetry_config,
)
from rsl_rl.utils import resolve_obs_groups, store_code_state

import Tactile_Lab
from pathlib import Path  # instead of `import pathlib as Path`

TACTILE_LAB_SRC = Path(Tactile_Lab.__file__).resolve().parent
_PARAMS_DIR = Path(TACTILE_LAB_SRC).joinpath("tasks/direct/obj_push_codesign/codesign_toolkit/Codesign_Assets/params")
reward_history_dir = Path(TACTILE_LAB_SRC).joinpath("tasks/direct/obj_push_codesign/codesign_toolkit/Codesign_Assets/reward_history")
params_path = _PARAMS_DIR.joinpath("current_params.json")

rewbuffer_path = reward_history_dir.joinpath("rewbuffer.pkl")
lenbuffer_path = reward_history_dir.joinpath("lenbuffer.pkl")
windowed_attempt_buffer_path = reward_history_dir.joinpath("windowed_attempt_buffer.pkl")
reward_history_path = reward_history_dir.joinpath("reward_history.json")
per_iteration_hardware_reward_history_path = reward_history_dir.joinpath(f"per_iteration_hardware_reward_history.json")
per_iteration_hardware_parameters_path = reward_history_dir.joinpath(f"per_iteration_hardware_parameters.json")

class OnPolicyCoDesignRunner:
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
        self.current_learning_iteration = 1
        self.git_status_repos = [rsl_rl.__file__]

    def learn(
        self,
        num_learning_iterations: int,
        hardware_iteration: int,
        noisy_iter_threshold: int,
        init_at_random_ep_len: bool = False,
    ) -> None:
        # Initialize writer
        self._prepare_logging_writer()

        # Randomize initial episode length per environment (for exploration) - for some reason this gives the same "random" values on every hardware iteration
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )

        # Start learning
        obs = self.env.get_observations().to(self.device)
        self.train_mode()  # switch to train mode (for dropout for example)

        # Book keeping
        ep_infos = []
        maxbufferlength = 512
        successwindowmaxbufferlength = 2048 # 768 512
        cnn_skip_iterations = 0  # freeze CNN encoder for this many iterations after morphology change
        policy_warmup_iterations = 30  # number of iterations to warmup the policy after morphology change before logging rewards and allowing saves (to avoid noise from initial performance drop)
        rew_skip_iterations = policy_warmup_iterations  # number of iterations to skip from reward buffer after task reset
        cnn_frozen = False
        per_it_success_ratio = 0.0  # track latest successes/attempts ratio for this hardware iteration
        per_it_success_ratio_buffer = deque(maxlen=maxbufferlength)  # buffer for success ratio logging per hardware iteration

        # load reward and length buffers from previous hardware iterations
        if hardware_iteration == 0:
            # fresh buffers for the first hardware iteration
            rewbuffer = deque(maxlen=maxbufferlength)
            lenbuffer = deque(maxlen=maxbufferlength)
            windowed_attempt_buffer = deque(maxlen=successwindowmaxbufferlength)  # 1=success, 0=non-success completion
            self.env.unwrapped.windowed_attempt_buffer = windowed_attempt_buffer  # share with env for _get_dones updates
            best_mean_reward = -float("inf")
            historical_success_ratio = float(0)
            historical_attempt_count = float(0)
            historical_success_count = float(0)
        else:
            with open(rewbuffer_path, "rb") as f:
                rewbuffer = pickle.load(f)
            with open(lenbuffer_path, "rb") as f:
                lenbuffer = pickle.load(f)
            with open(windowed_attempt_buffer_path, "rb") as f:
                windowed_attempt_buffer = pickle.load(f)
            self.env.unwrapped.windowed_attempt_buffer = windowed_attempt_buffer  # share with env for _get_dones updates
            with open(reward_history_path, "r", encoding="utf-8") as f:
                reward_history_dict = json.load(f)
            best_mean_reward = reward_history_dict.get("best_mean_reward", -float("inf"))

        per_it_rewbuffer = deque(maxlen=maxbufferlength)  # buffer for current hardware iteration for per-iteration logging

        # -------- Load current hardware design parameters --------
        with open(params_path, "r", encoding="utf-8") as f:
                    params = json.load(f)

        params_dict = {
            "base_sphere_centre_z_offset": float(params["base_sphere_centre_z_offset"]),
            "concave_dimple_diameter": float(params["concave_dimple_diameter"]),
            "convex_dimple_diameter": float(params["convex_dimple_diameter"]),
            "concave_dimple_depth_scale": float(params["concave_dimple_depth_scale"]),
            "convex_dimple_height_scale": float(params["convex_dimple_height_scale"]),
        }

        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        # --- Best policy tracking ---
        
        best_model_path = os.path.join(self.log_dir, "best_model.pt") if self.log_dir else None

        # Create buffers for logging extrinsic and intrinsic rewards
        if self.alg.rnd:
            erewbuffer = deque(maxlen=maxbufferlength)
            irewbuffer = deque(maxlen=maxbufferlength)
            cur_ereward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
            cur_ireward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        # Ensure all parameters are in-synced
        if self.is_distributed:
            print(f"Synchronizing parameters for rank {self.gpu_global_rank}...")
            self.alg.broadcast_parameters()

        # Start training
        if hardware_iteration == 0:
            start_iter = self.current_learning_iteration
        else:
            start_iter = self.current_learning_iteration + 1  # continue from last iteration
        tot_iter = start_iter + num_learning_iterations - 1

        for it in range(start_iter, tot_iter + 1):
            current_iter = it - start_iter + 1

            # Freeze CNN encoder during morphology transition to protect learned representations
            if cnn_skip_iterations > 0 :
                if current_iter == 1 and not cnn_frozen:
                    self._set_cnn_requires_grad(False)
                    cnn_frozen = True
                    print(f"[CNN Freeze] Freezing CNN encoder for {cnn_skip_iterations} iterations after morphology change.")

                # Unfreeze CNN encoder after skip iterations
                if current_iter == cnn_skip_iterations + 1 and cnn_frozen:
                    self._set_cnn_requires_grad(True)
                    cnn_frozen = False
                    print(f"[CNN Freeze] Unfreezing CNN encoder at iteration {it}.")

            start = time.time()
            # Rollout
            with torch.inference_mode():
                for _ in range(self.num_steps_per_env):
                    # Sample actions
                    actions = self.alg.act(obs)
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
                        if current_iter > rew_skip_iterations: # skip first data points from reward buffer after task reset
                            rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist()) # skip first data points
                            lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                            per_it_rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
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

            # Collect success ratio for per-iteration logging if available
            if "per_it_success_ratio" in extras["log"]:
                success_val = extras["log"]["per_it_success_ratio"]
                per_it_success_ratio = float(success_val)

            if "historical_success_ratio" in extras["log"]:
                historical_ratio_val = extras["log"]["historical_success_ratio"]
                historical_attempt_val = extras["log"]["historical_attempt_count"]
                historical_success_val = extras["log"]["historical_success_count"]
                historical_success_ratio = float(historical_ratio_val)
                historical_attempt_count = float(historical_attempt_val)
                historical_success_count = float(historical_success_val)

            if current_iter > policy_warmup_iterations: # skip logging rewards and saving models until after policy warmup iterations to avoid noise from initial performance drop after morphology change
                # Update policy
                loss_dict = self.alg.update()

            stop = time.time()
            learn_time = stop - start
            self.current_learning_iteration = it
            
            # # Given that our hardware iterations are short (W.R.T. typical RL training), we ignore the first episode reward(s) after a task reset to avoid invalid/biased data due to environment reset.
            # if current_iter > skip_iterations:

            if self.log_dir is not None and not self.disable_logs and current_iter > policy_warmup_iterations: # skip logging rewards and saving models until after policy warmup iterations to avoid noise from initial performance drop after morphology change

                # Log information
                self.log(locals())

                # -------- Save best model --------
                # no best model in first hardware iteration to avoid noise before any
                # complete episodes and ensure >1 episode is completed in the set.
                # The iteration threshold is configurable via noisy_iter_threshold.
                if hardware_iteration > 0 and len(rewbuffer) > 0 and it > noisy_iter_threshold:
                    
                    mean_rew = statistics.mean(rewbuffer)

                    if mean_rew > best_mean_reward:
                        best_mean_reward = mean_rew
                        _windowed_sr = sum(windowed_attempt_buffer) / len(windowed_attempt_buffer) if len(windowed_attempt_buffer) > 0 else 0.0
                        self._log_best_model(
                            it, hardware_iteration, mean_rew, params_dict,
                            per_it_success_ratio, historical_success_ratio,
                            _windowed_sr, best_model_path,
                        )

                    # Save model
                    if it % self.save_interval == 0:
                        self.save(os.path.join(self.log_dir, f"model_{it}.pt"))

            else:
                print(f"Learning iteration: {it}/{tot_iter}\n Initial buffering phase before reaching stable reward data...")
                self.alg.storage.clear()

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

        print(f"Logging Hardware Iteration {hardware_iteration} complete. Best mean reward: {best_mean_reward:.3f}")

        # Compute windowed success ratio for logging
        windowed_success_ratio = sum(windowed_attempt_buffer) / len(windowed_attempt_buffer) if len(windowed_attempt_buffer) > 0 else 0.0

        # Save the final model after training
        if self.log_dir is not None and not self.disable_logs:
            self.save(os.path.join(self.log_dir, f"model_{self.current_learning_iteration}.pt"))

        self._log_hardware_iteration_end(
            hardware_iteration, params_dict, rewbuffer, lenbuffer,
            windowed_attempt_buffer,
            best_mean_reward, historical_success_ratio, historical_attempt_count,
            historical_success_count, per_it_rewbuffer, per_it_success_ratio,
        )


    # --------------- Codesign-specific logging helpers ---------------

    def _log_best_model(
        self,
        it: int,
        hardware_iteration: int,
        mean_rew: float,
        params_dict: dict,
        per_it_success_ratio: float,
        historical_success_ratio: float,
        windowed_success_ratio: float,
        best_model_path: str,
    ) -> None:
        """Save the new best model checkpoint and log associated design parameters."""
        print(
            f"\033[92m[Best Model] Iter {it}: mean reward improved to {mean_rew:.3f}, saving model.\033[0m"
        )
        self.save(best_model_path, hardware_iteration, params_dict)

        # Plain-text summary
        best_log_path = os.path.join(self.log_dir, "best_policy_plus_design_params.txt")
        with open(best_log_path, "w") as f:
            f.write(
                f"Iter {it}: mean_reward = {mean_rew:.6f}\n"
                f" Hardware Iteration = {hardware_iteration}\n"
                f" Hardware Params = {params_dict}"
            )

        # Structured JSON (latest best only)
        details = {
            "policy_iteration": it,
            "current_best_mean_reward": float(mean_rew),
            "hardware_iteration": hardware_iteration,
            "per_it_success_ratio": per_it_success_ratio,
            "historical_success_ratio": historical_success_ratio,
            "windowed_success_ratio": windowed_success_ratio,
            **params_dict,
        }
        best_log_json_path = os.path.join(self.log_dir, "best_policy_plus_design_params.json")
        with open(best_log_json_path, "w", encoding="utf-8") as f:
            json.dump(details, f, indent=2)

        # Append to cumulative history
        full_best_log_json_path = os.path.join(self.log_dir, "all_best_policy_plus_design_params.json")
        with open(full_best_log_json_path, "a", encoding="utf-8") as f:
            json.dump(details, f, indent=2)
            f.write("\n")

    def _log_hardware_iteration_end(
        self,
        hardware_iteration: int,
        params_dict: dict,
        rewbuffer: deque,
        lenbuffer: deque,
        windowed_attempt_buffer: deque,
        best_mean_reward: float,
        historical_success_ratio: float,
        historical_attempt_count: float,
        historical_success_count: float,
        per_it_rewbuffer: deque,
        per_it_success_ratio: float,
    ) -> None:
        """Serialise reward buffers, hardware parameters, and per-iteration history at the end of a hardware iteration."""
        # Compute windowed success ratio
        windowed_success_ratio = sum(windowed_attempt_buffer) / len(windowed_attempt_buffer) if len(windowed_attempt_buffer) > 0 else 0.0

        # ---- Accumulate hardware parameters into a single JSON file ----
        if per_iteration_hardware_parameters_path.exists():
            try:
                with open(per_iteration_hardware_parameters_path, "r", encoding="utf-8") as f:
                    hw_params_history = json.load(f)
            except json.JSONDecodeError:
                hw_params_history = {}
        else:
            hw_params_history = {}

        prefix = f"hardware_iteration_{hardware_iteration}"
        hw_params_history[f"{prefix}_z"] = float(params_dict["base_sphere_centre_z_offset"])
        hw_params_history[f"{prefix}_cd"] = float(params_dict["concave_dimple_diameter"])
        hw_params_history[f"{prefix}_vd"] = float(params_dict["convex_dimple_diameter"])
        hw_params_history[f"{prefix}_cds"] = float(params_dict["concave_dimple_depth_scale"])
        hw_params_history[f"{prefix}_vhs"] = float(params_dict["convex_dimple_height_scale"])

        with open(per_iteration_hardware_parameters_path, "w", encoding="utf-8") as f:
            json.dump(hw_params_history, f, indent=2)
        if self.log_dir is not None:
            shutil.copy2(per_iteration_hardware_parameters_path, os.path.join(self.log_dir, "per_iteration_hardware_parameters.json"))

        # ---- Persist reward and length buffers for the next hardware iteration ----
        with open(rewbuffer_path, "wb") as f:
            pickle.dump(rewbuffer, f)
        with open(lenbuffer_path, "wb") as f:
            pickle.dump(lenbuffer, f)
        with open(windowed_attempt_buffer_path, "wb") as f:
            pickle.dump(windowed_attempt_buffer, f)

        # Persist cross-iteration reward summary
        reward_history_dict = {
            "best_mean_reward": float(best_mean_reward),
            "historical_success_ratio": float(historical_success_ratio),
            "historical_attempt_count": float(historical_attempt_count),
            "historical_success_count": float(historical_success_count),
        }
        with open(reward_history_path, "w", encoding="utf-8") as f:
            json.dump(reward_history_dict, f, indent=2)

        # Accumulate per-hardware-iteration reward history
        per_it_mean_rew = statistics.mean(per_it_rewbuffer)
        per_it_final_success_ratio = float(per_it_success_ratio)

        if per_iteration_hardware_reward_history_path.exists():
            try:
                with open(per_iteration_hardware_reward_history_path, "r", encoding="utf-8") as f:
                    per_it_history = json.load(f)
            except json.JSONDecodeError:
                per_it_history = {}
        else:
            per_it_history = {}

        prefix = f"hardware_iteration_{hardware_iteration}"
        per_it_history[f"{prefix}_mean_reward"] = float(per_it_mean_rew)
        per_it_history[f"{prefix}_success_ratio"] = float(per_it_final_success_ratio)
        per_it_history[f"{prefix}_historical_success_ratio"] = float(historical_success_ratio)
        per_it_history[f"{prefix}_windowed_success_ratio"] = float(windowed_success_ratio)
        per_it_history[f"{prefix}_params"] = (
            f"z={params_dict['base_sphere_centre_z_offset']:.3f} "
            f"cd={params_dict['concave_dimple_diameter']:.3f} "
            f"vd={params_dict['convex_dimple_diameter']:.3f} "
            f"cds={params_dict['concave_dimple_depth_scale']:.3f} "
            f"vhs={params_dict['convex_dimple_height_scale']:.3f}"
        )

        with open(per_iteration_hardware_reward_history_path, "w", encoding="utf-8") as f:
            json.dump(per_it_history, f, indent=2)
        if self.log_dir is not None:
            shutil.copy2(per_iteration_hardware_reward_history_path, os.path.join(self.log_dir, "per_iteration_hardware_reward_history.json"))

        # ---- Log hardware-iteration-level scalars to TensorBoard ----
        if self.writer is not None:
            step = hardware_iteration
            self.writer.add_scalar("Codesign/mean_reward", float(per_it_mean_rew), step)
            self.writer.add_scalar("Codesign/best_mean_reward", float(best_mean_reward), step)
            self.writer.add_scalar("Codesign/success_ratio", float(per_it_final_success_ratio), step)
            self.writer.add_scalar("Codesign/historical_success_ratio", float(historical_success_ratio), step)
            self.writer.add_scalar("Codesign/historical_attempt_count", float(historical_attempt_count), step)
            self.writer.add_scalar("Codesign/historical_success_count", float(historical_success_count), step)
            # Log windowed success ratio
            self.writer.add_scalar("Codesign/windowed_success_ratio", float(windowed_success_ratio), step)
            # Log each morphology parameter
            for param_name, param_val in params_dict.items():
                self.writer.add_scalar(f"Codesign_Params/{param_name}", float(param_val), step)
            self.writer.flush()

    # --------------- General logging ---------------

    def log(self, locs: dict, width: int = 80, pad: int = 35) -> None:
        # Compute the collection size
        collection_size = self.num_steps_per_env * self.env.num_envs * self.gpu_world_size
        # Update total time-steps and time
        self.tot_timesteps += collection_size
        self.tot_time += locs["collection_time"] + locs["learn_time"]
        iteration_time = locs["collection_time"] + locs["learn_time"]

        # Log episode information
        ep_string = ""
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
                    ep_string += f"""{f"{key}:":>{pad}} {value:.4f}\n"""
                else:
                    self.writer.add_scalar("Episode/" + key, value, locs["it"])
                    ep_string += f"""{f"Mean {key}:":>{pad}} {value:.4f}\n"""

        mean_std = self.alg.policy.action_std.mean()
        fps = int(collection_size / (locs["collection_time"] + locs["learn_time"]))

        # Log losses
        for key, value in locs["loss_dict"].items():
            self.writer.add_scalar(f"Loss/{key}", value, locs["it"])
        self.writer.add_scalar("Loss/learning_rate", self.alg.learning_rate, locs["it"])

        # Log noise std
        self.writer.add_scalar("Policy/mean_noise_std", mean_std.item(), locs["it"])

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
                f"""{"Mean action noise std:":>{pad}} {mean_std.item():.2f}\n"""
            )
            # Print losses
            for key, value in locs["loss_dict"].items():
                log_string += f"""{f"Mean {key} loss:":>{pad}} {value:.4f}\n"""
            # Print rewards
            if hasattr(self.alg, "rnd") and self.alg.rnd:
                log_string += (
                    f"""{"Mean extrinsic reward:":>{pad}} {statistics.mean(locs["erewbuffer"]):.2f}\n"""
                    f"""{"Mean intrinsic reward:":>{pad}} {statistics.mean(locs["irewbuffer"]):.2f}\n"""
                )
            log_string += f"""{"Mean reward:":>{pad}} {statistics.mean(locs["rewbuffer"]):.2f}\n"""
            # Print episode information
            log_string += f"""{"Mean episode length:":>{pad}} {statistics.mean(locs["lenbuffer"]):.2f}\n"""
            # Add header for logged metrics
            log_string += f"""\n{'Mean values across this policy it:':>{pad}}\n"""
        else:
            log_string = (
                f"""{"#" * width}\n"""
                f"""{str.center(width, " ")}\n\n"""
                f"""{"Computation:":>{pad}} {fps:.0f} steps/s (collection: {locs["collection_time"]:.3f}s, learning {
                    locs["learn_time"]:.3f}s)\n"""
                f"""{"Mean action noise std:":>{pad}} {mean_std.item():.2f}\n"""
            )
            for key, value in locs["loss_dict"].items():
                log_string += f"""{f"{key}:":>{pad}} {value:.4f}\n"""
            # Add header for logged metrics
            log_string += f"""\n{'Mean values across this policy it:':>{pad}}\n"""

        log_string += ep_string
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

    def save(self, path: str, hardware_iteration: int = 0, current_params_dict: dict | None = None, infos: dict | None = None) -> None:
        # Save model
        saved_dict = {
            "model_state_dict": self.alg.policy.state_dict(),
            "optimizer_state_dict": self.alg.optimizer.state_dict(),
            "iter": self.current_learning_iteration,
            "hardware_iteration": hardware_iteration,
            "hardware_params": current_params_dict,
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

    def _set_cnn_requires_grad(self, requires_grad: bool) -> None:
        """Freeze or unfreeze CNN encoder parameters in the actor-critic policy.

        This protects learned visual representations during the initial
        buffering phase after a morphology change, where observations may
        be noisy or misleading.
        """
        policy = self.alg.policy
        for attr in ("actor_cnns", "critic_cnns"):
            cnns = getattr(policy, attr, None)
            if cnns is not None:
                for param in cnns.parameters():
                    param.requires_grad = requires_grad

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
        actor_critic: ActorCritic | ActorCriticRecurrent | ActorCriticCNN = actor_critic_class(
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

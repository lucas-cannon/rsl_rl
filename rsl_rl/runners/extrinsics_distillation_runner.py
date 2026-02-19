import os
import time
import statistics
import torch
from collections import deque
from tensordict import TensorDict

import rsl_rl
from rsl_rl.env import VecEnv
from rsl_rl.utils import resolve_obs_groups, store_code_state
from rsl_rl.algorithms import ExtrinsicsDistillation
from rsl_rl.modules import StudentTeacherExtrinsics, StudentTeacherMultiModalExtrinsics
from rsl_rl.runners import OnPolicyRunner


class ExtrinsicsDistillationRunner(OnPolicyRunner):
    """
    Runner for Stage-2 extrinsics distillation.
    """

    def __init__(self, env: VecEnv, train_cfg: dict, log_dir: str | None = None, device: str = "cpu"):

        self.cfg = train_cfg
        self.alg_cfg = train_cfg["algorithm"]
        self.policy_cfg = train_cfg["policy"]

        self.device = device
        self.env = env
        self.log_dir = log_dir

        self._configure_multi_gpu()
        
        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]

        # Resolve observation groups
        obs = self.env.get_observations()
        self.cfg["obs_groups"] = resolve_obs_groups(
            obs,
            self.cfg["obs_groups"],
            default_sets=["priv", "obs", "obs_hist"]
        )

        # Construct algorithm
        self.alg = self._construct_algorithm(obs)

        self.disable_logs = self.is_distributed and self.gpu_global_rank != 0

        self.log_dir = log_dir
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0
        self.git_status_repos = [rsl_rl.__file__]

    # LEARN
    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False) -> None:
        # Initialize writer
        self._prepare_logging_writer()
        # Check if teacher is loaded
        if not self.alg.policy.loaded_teacher:
            raise ValueError("Teacher model parameters not loaded. Please load a teacher model to distill.")

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
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        # --- Best policy tracking ---
        best_mean_reward = -float("inf")
        best_bc_loss = float("inf")  
        best_model_path = os.path.join(self.log_dir, "best_model.pt") if self.log_dir else None

        # Ensure all parameters are in-synced
        if self.is_distributed:
            print(f"Synchronizing parameters for rank {self.gpu_global_rank}...")
            self.alg.broadcast_parameters()

        # Start training
        start_iter = self.current_learning_iteration
        tot_iter = start_iter + num_learning_iterations
        for it in range(start_iter, tot_iter):
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
                    # Book keeping
                    if self.log_dir is not None:
                        if "episode" in extras:
                            ep_infos.append(extras["episode"])
                        elif "log" in extras:
                            ep_infos.append(extras["log"])
                        # Update rewards
                        cur_reward_sum += rewards
                        # Update episode length
                        cur_episode_length += 1
                        # Clear data for completed episodes
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0

                stop = time.time()
                collection_time = stop - start
                start = stop

            # Update policy
            loss_dict = self.alg.update()

            stop = time.time()
            learn_time = stop - start
            self.current_learning_iteration = it

            if self.log_dir is not None and not self.disable_logs:
                # Log information
                self.log(locals())
                # set_trace()
                if "extrinsics_mse" in loss_dict:

                    bc_loss = float(loss_dict["extrinsics_mse"])
                    if it > 100:   # avoid noise before any complete episodes
                        if bc_loss < best_bc_loss:
                            best_bc_loss = bc_loss
                            print(
                                f"\033[92m[Best Model] Iter {it}: bc_loss improved to {bc_loss:.6f}, saving model.\033[0m"
                            )
                            self.save(best_model_path)

                            # Write to logging text file
                            best_log_path = os.path.join(self.log_dir, "best_policy.txt")
                            with open(best_log_path, "a") as f:
                                f.write(f"Iter {it}: bc_loss = {bc_loss:.6f}\n")

                # ========== PPO mode: track reward ==========
                else:
                    if it > 100:  # avoid noise in early iterations
                        mean_rew = statistics.mean(rewbuffer)

                        if mean_rew > best_mean_reward:
                            best_mean_reward = mean_rew
                            print(
                                f"\033[92m[Best Model] Iter {it}: mean reward improved to {mean_rew:.3f}, saving model.\033[0m"
                            )
                            self.save(best_model_path)

                # Save every N iterations
                if it % self.save_interval == 0:
                    self.save(os.path.join(self.log_dir, f"model_{it}.pt"))

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

        # Save the final model after training
        if self.log_dir is not None and not self.disable_logs:
            self.save(os.path.join(self.log_dir, f"model_{self.current_learning_iteration}.pt"))

    def _construct_algorithm(self, obs: TensorDict):

        policy_cfg = self.policy_cfg.copy()
        alg_cfg = self.alg_cfg.copy()

        student_teacher_class = eval(policy_cfg.pop("class_name"))
        student_teacher = student_teacher_class(
            obs,
            self.cfg["obs_groups"],
            self.env.num_actions,
            **policy_cfg,
        ).to(self.device)

        alg_class = eval(alg_cfg.pop("class_name"))
        alg: ExtrinsicsDistillation = alg_class(
            student_teacher,
            device=self.device,
            **alg_cfg,
            multi_gpu_cfg=self.multi_gpu_cfg,
        )

        alg.init_storage(
            "extrinsics_distillation",
            self.env.num_envs,
            self.num_steps_per_env,
            obs,
            [self.env.num_actions],
        )

        return alg

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
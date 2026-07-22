from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from rsl_rl.runners.on_policy_runner import OnPolicyRunner


class _Policy:
    action_std = torch.ones(1)
    entropy = torch.tensor([0.5])


class _Algorithm:
    rnd = False
    policy = _Policy()

    def act(self, observations):
        return torch.zeros(1, 1)

    def process_env_step(self, observations, rewards, dones, extras):
        pass

    def compute_returns(self, observations):
        pass

    def update(self):
        return {}


class _Environment:
    num_envs = 1
    device = "cpu"
    max_episode_length = 10

    def __init__(self):
        self.episode_length_buf = torch.zeros(1, dtype=torch.long)
        self.step_index = 0

    def get_observations(self):
        return torch.zeros(1, 1)

    def step(self, actions):
        self.step_index += 1
        return (
            torch.zeros(1, 1),
            torch.tensor([float(self.step_index)]),
            torch.tensor([True]),
            {"time_outs": torch.tensor([False])},
        )


def _fake_runner(log_dir: str, current_iteration: int = 0):
    runner = object.__new__(OnPolicyRunner)
    runner.env = _Environment()
    runner.alg = _Algorithm()
    runner.device = "cpu"
    runner.num_steps_per_env = 1
    runner.gpu_world_size = 1
    runner.is_distributed = False
    runner.gpu_global_rank = 0
    runner.current_learning_iteration = current_iteration
    runner.log_dir = log_dir
    runner.disable_logs = False
    runner.save_interval = 1_000
    runner.git_status_repos = []
    runner.logger_type = "tensorboard"
    runner._prepare_logging_writer = lambda: None
    runner.train_mode = lambda: None
    runner.log = lambda *_args, **_kwargs: None
    return runner


class IterationCallbackTests(unittest.TestCase):
    def test_callback_keeps_rolling_rewards_and_stops_cleanly(self):
        with tempfile.TemporaryDirectory() as folder:
            runner = _fake_runner(folder)
            saved = []
            runner.save = saved.append
            metrics = []
            runner.learn(
                10,
                iteration_callback=lambda row: metrics.append(row) or row["iteration"] == 2,
            )
            self.assertEqual([row["iteration"] for row in metrics], [0, 1, 2])
            self.assertEqual(
                [row["mean_reward"] for row in metrics], [1.0, 1.5, 2.0]
            )
            self.assertEqual(metrics[-1]["best_mean_reward"], 2.0)
            self.assertEqual(runner.current_learning_iteration, 2)
            self.assertEqual(
                sum(Path(path).name == "best_model.pt" for path in saved), 3
            )

    def test_nonzero_start_advances_without_replaying(self):
        with tempfile.TemporaryDirectory() as folder:
            runner = _fake_runner(folder, current_iteration=500)
            runner.save = lambda _path: None
            iterations = []
            runner.learn(2, iteration_callback=lambda row: iterations.append(row["iteration"]))
            self.assertEqual(iterations, [500, 501])
            self.assertEqual(runner.current_learning_iteration, 501)


if __name__ == "__main__":
    unittest.main()

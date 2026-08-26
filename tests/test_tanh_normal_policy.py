import math

import torch
from tensordict import TensorDict

from rsl_rl.modules import ActorCritic


def _policy(num_actions: int = 4, action_distribution: str = "tanh_normal") -> ActorCritic:
    obs = TensorDict(
        {"policy": torch.zeros(8, 3), "critic": torch.zeros(8, 5)},
        batch_size=[8],
    )
    return ActorCritic(
        obs,
        {"policy": ["policy"], "critic": ["critic"]},
        num_actions,
        actor_hidden_dims=[8],
        critic_hidden_dims=[8],
        init_noise_std=0.8,
        noise_std_type="log",
        action_distribution=action_distribution,
    )


def test_tanh_normal_samples_and_inference_are_bounded():
    policy = _policy()
    obs = TensorDict(
        {"policy": torch.randn(8, 3), "critic": torch.randn(8, 5)},
        batch_size=[8],
    )

    actions = policy.act(obs)

    assert torch.all(actions.abs() < 1.0)
    assert torch.all(policy.act_inference(obs).abs() < 1.0)
    assert torch.isfinite(policy.get_actions_log_prob(actions)).all()
    assert torch.isfinite(policy.entropy).all()


def test_tanh_normal_log_prob_matches_change_of_variables():
    policy = _policy(num_actions=2)
    obs = TensorDict(
        {"policy": torch.zeros(8, 3), "critic": torch.zeros(8, 5)},
        batch_size=[8],
    )
    actions = policy.act(obs)
    latent = policy.latent_action

    expected = (
        policy.distribution.log_prob(latent)
        - torch.log1p(-actions.square())
    ).sum(dim=-1)

    torch.testing.assert_close(policy.get_actions_log_prob(actions), expected)


def test_normal_distribution_remains_backward_compatible():
    policy = _policy(action_distribution="normal")
    obs = TensorDict(
        {"policy": torch.zeros(8, 3), "critic": torch.zeros(8, 5)},
        batch_size=[8],
    )
    actions = policy.act(obs)

    torch.testing.assert_close(actions, policy.latent_action)
    torch.testing.assert_close(
        policy.entropy,
        policy.distribution.entropy().sum(dim=-1),
    )
    assert math.isfinite(policy.entropy.mean().item())

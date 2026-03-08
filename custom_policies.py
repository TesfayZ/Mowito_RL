"""
Custom policy classes for RL stabilization experiments.

Contains:
1. **RWAI (Reward-Range-Aware Initialization)**: Two variants that initialize
   critic last-layer biases/weights based on expected Q-value ranges.

2. **Adaptive Gradient Scaling**: Tracks running mean/std of actor pre-activations
   and scales them to a target range before tanh, improving gradient flow in
   the saturated regions of tanh.

These modifications are orthogonal and compose cleanly:
- RWAI modifies critic initialization
- Adaptive scaling modifies actor forward pass
"""

import torch as th
import torch.nn as nn

from gymnasium import spaces
from stable_baselines3.common.policies import ContinuousCritic
from stable_baselines3.common.preprocessing import get_action_dim
from stable_baselines3.common.torch_layers import create_mlp
from stable_baselines3.sac import policies as sac_policies
from stable_baselines3.sac.policies import SACPolicy
from stable_baselines3.td3 import policies as td3_policies
from stable_baselines3.td3.policies import TD3Policy


def compute_q_range(
    r_min: float,
    r_max: float,
    gamma: float,
    horizon: int,
    reward_type: str = "dense",
):
    """Compute the expected Q-value range from reward bounds.

    Args:
        r_min: Minimum per-step reward.
        r_max: Maximum per-step reward.
        gamma: Discount factor.
        horizon: Episode length (max timesteps).
        reward_type: "dense" (reward every step) or "sparse" (reward at end).

    Returns:
        (q_min, q_max): Tuple of expected Q-value bounds.
    """
    if reward_type == "dense":
        geo_sum = (1.0 - gamma ** horizon) / (1.0 - gamma)
        q_min = r_min * geo_sum
        q_max = r_max * geo_sum
    elif reward_type == "sparse":
        q_min = r_min * gamma ** (horizon - 1)
        q_max = r_max * gamma ** (horizon - 1)
    else:
        raise ValueError(f"Unknown reward_type: {reward_type}")
    return q_min, q_max


def init_last_layer_for_reward_range(
    layer: nn.Linear,
    q_min: float,
    q_max: float,
):
    """Initialize a linear layer's bias for an expected Q-value range (v1).

    Shifts only the bias to Q_mid = (q_min + q_max) / 2. The weights are
    left at their default initialization, preserving gradient flow and
    state differentiation.
    """
    q_mid = (q_min + q_max) / 2.0
    if layer.bias is not None:
        nn.init.constant_(layer.bias, q_mid)


def init_last_layer_scaled(q_net: nn.Sequential, q_min: float, q_max: float):
    """Initialize the last layer so outputs span [q_min, q_max] (v2).

    Uses empirical calibration: passes random inputs through the hidden layers
    to measure the actual output distribution, then scales the last-layer
    weights so that output std ≈ Q_range / 4 (±2σ covers [q_min, q_max]).

    After scaling, the bias is recalibrated so the mean output equals Q_mid.
    This compensates for the non-zero mean contribution from scaled weights
    acting on non-negative ReLU activations.

    Args:
        q_net: The full Q-network (nn.Sequential of Linear-ReLU-...-Linear).
        q_min: Minimum expected cumulative return.
        q_max: Maximum expected cumulative return.
    """
    q_mid = (q_min + q_max) / 2.0
    q_range = q_max - q_min
    target_std = q_range / 4.0  # ±2σ covers [q_min, q_max]

    last_layer = q_net[-1]
    hidden = q_net[:-1]

    # Step 1: Measure current output std
    input_dim = q_net[0].in_features  # obs_dim + action_dim
    with th.no_grad():
        x = th.randn(2048, input_dim, device=last_layer.weight.device)
        h = hidden(x)
        out = last_layer(h)
        current_std = out.std().item()

    # Step 2: Scale weights to achieve desired output std
    # Cap at 100x to prevent extreme gradient amplification in early training
    scale = 1.0
    if current_std > 1e-8:
        scale = min(target_std / current_std, 100.0)
        last_layer.weight.data *= scale

    # Step 3: Recalibrate bias so mean output = Q_mid
    # After scaling, the weighted sum W@h has a non-zero mean (because
    # h ≥ 0 from ReLU and scaling amplifies any asymmetry). We measure
    # this mean and set the bias to compensate.
    with th.no_grad():
        nn.init.constant_(last_layer.bias, 0.0)
        h = hidden(th.randn(2048, input_dim, device=last_layer.weight.device))
        weighted_mean = last_layer(h).mean().item()

    bias_value = q_mid - weighted_mean
    nn.init.constant_(last_layer.bias, bias_value)

    print(f"    RWAI v2: weights scaled {scale:.0f}x, "
          f"bias={bias_value:.1f} (Q_mid={q_mid:.1f}), "
          f"output std: {current_std:.4f} → {target_std:.1f}")


# ── V1: Bias-only RWAI ──────────────────────────────────────────────────────


class RewardAwareContinuousCritic(ContinuousCritic):
    """Critic with bias-only RWAI (v1)."""

    def __init__(self, *args, q_min: float = 0.0, q_max: float = 500.0, **kwargs):
        super().__init__(*args, **kwargs)
        for q_net in self.q_networks:
            last_layer = q_net[-1]
            init_last_layer_for_reward_range(last_layer, q_min, q_max)


class RewardAwareSACPolicy(SACPolicy):
    """SAC policy with bias-only RWAI (v1)."""

    def __init__(self, *args, q_min: float = 0.0, q_max: float = 500.0, **kwargs):
        self._q_min = q_min
        self._q_max = q_max
        super().__init__(*args, **kwargs)

    def make_critic(self, features_extractor=None):
        critic_kwargs = self._update_features_extractor(
            self.critic_kwargs, features_extractor
        )
        return RewardAwareContinuousCritic(
            q_min=self._q_min,
            q_max=self._q_max,
            **critic_kwargs,
        ).to(self.device)


class RewardAwareTD3Policy(TD3Policy):
    """TD3/DDPG policy with bias-only RWAI (v1)."""

    def __init__(self, *args, q_min: float = 0.0, q_max: float = 500.0, **kwargs):
        self._q_min = q_min
        self._q_max = q_max
        super().__init__(*args, **kwargs)

    def make_critic(self, features_extractor=None):
        critic_kwargs = self._update_features_extractor(
            self.critic_kwargs, features_extractor
        )
        return RewardAwareContinuousCritic(
            q_min=self._q_min,
            q_max=self._q_max,
            **critic_kwargs,
        ).to(self.device)


# ── V2: Scaled RWAI (empirically calibrated) ────────────────────────────────


class ScaledRewardAwareContinuousCritic(ContinuousCritic):
    """Critic with output range empirically calibrated to [q_min, q_max].

    Scales last-layer weights so that the output std spans the Q-value range,
    giving different state-action pairs meaningfully different initial
    Q-estimates. Uses empirical forward pass for accurate calibration.
    """

    def __init__(self, *args, q_min: float = 0.0, q_max: float = 500.0, **kwargs):
        super().__init__(*args, **kwargs)
        for q_net in self.q_networks:
            init_last_layer_scaled(q_net, q_min, q_max)


class ScaledRewardAwareSACPolicy(SACPolicy):
    """SAC policy with scaled RWAI (v2)."""

    def __init__(self, *args, q_min: float = 0.0, q_max: float = 500.0, **kwargs):
        self._q_min = q_min
        self._q_max = q_max
        super().__init__(*args, **kwargs)

    def make_critic(self, features_extractor=None):
        critic_kwargs = self._update_features_extractor(
            self.critic_kwargs, features_extractor
        )
        return ScaledRewardAwareContinuousCritic(
            q_min=self._q_min,
            q_max=self._q_max,
            **critic_kwargs,
        ).to(self.device)


class ScaledRewardAwareTD3Policy(TD3Policy):
    """TD3/DDPG policy with scaled RWAI (v2). Works for both TD3 and DDPG."""

    def __init__(self, *args, q_min: float = 0.0, q_max: float = 500.0, **kwargs):
        self._q_min = q_min
        self._q_max = q_max
        super().__init__(*args, **kwargs)

    def make_critic(self, features_extractor=None):
        critic_kwargs = self._update_features_extractor(
            self.critic_kwargs, features_extractor
        )
        return ScaledRewardAwareContinuousCritic(
            q_min=self._q_min,
            q_max=self._q_max,
            **critic_kwargs,
        ).to(self.device)


# ── Adaptive Gradient Scaling ────────────────────────────────────────────────
# Tracks running min/max of pre-tanh activations and rescales them to a target
# range before tanh. This keeps activations in the high-gradient region of tanh,
# avoiding saturation-induced vanishing gradients.
#
# The scaler is a buffer-based module (no learnable parameters). It passes
# through unscaled until the first training batch initializes the stats.
# This is safe because learning_starts=2000 means the actor isn't trained
# for the first 2000 environment steps.


class AdaptiveGradientScaler(nn.Module):
    """Scale pre-activation values to [-target_range, +target_range] before tanh.

    Tracks running mean and standard deviation of inputs per action dimension
    (similar to BatchNorm) and linearly maps [mean - k*std, mean + k*std] to
    [-target_range, +target_range]. This is more principled than tracking
    min/max, which systematically underestimates the true range via EMA.

    Uses persistent buffers so stats survive model saving/loading but are NOT
    affected by polyak_update (which only syncs parameters).

    Args:
        action_dim: Number of action dimensions.
        target_range: Target output range magnitude (default 2.0 -> maps to [-2, 2]).
        k_std: Number of standard deviations to map to target_range (default 2.5).
    """

    def __init__(self, action_dim: int, target_range: float = 2.0, k_std: float = 2.5):
        super().__init__()
        self.target_range = target_range
        self.k_std = k_std
        # Running statistics (sentinel: g_var < 0 means uninitialized)
        self.register_buffer("g_mean", th.zeros(action_dim))
        self.register_buffer("g_var", th.full((action_dim,), -1.0))

    @property
    def initialized(self) -> bool:
        """True once we've seen at least one batch of data."""
        return not (self.g_var < 0).any()

    def forward(self, x: th.Tensor) -> th.Tensor:
        if self.training and x.shape[0] > 1:
            batch_mean = x.detach().mean(dim=0)
            batch_var = x.detach().var(dim=0)
            if self.initialized:
                # Exponential moving average with momentum 0.99
                self.g_mean.copy_(0.99 * self.g_mean + 0.01 * batch_mean)
                self.g_var.copy_(0.99 * self.g_var + 0.01 * batch_var)
            else:
                # First batch: initialize directly
                self.g_mean.copy_(batch_mean)
                self.g_var.copy_(batch_var)

        if not self.initialized:
            return x

        # Scale: map [mean - k*std, mean + k*std] -> [-target_range, +target_range]
        std = self.g_var.clamp(min=1e-6).sqrt()
        half_range = self.k_std * std
        return (x - self.g_mean) / half_range * self.target_range


# ── Custom TD3/DDPG Actors with Adaptive Scaling ────────────────────────────


class AdaptiveScalingTD3Actor(td3_policies.Actor):
    """TD3/DDPG actor with adaptive gradient scaling before tanh.

    Replaces the final nn.Tanh() in self.mu with AdaptiveGradientScaler → nn.Tanh().
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        action_dim = get_action_dim(self.action_space)

        # Rebuild self.mu: replace trailing Tanh with Scaler → Tanh
        layers = list(self.mu.children())
        if isinstance(layers[-1], nn.Tanh):
            layers[-1] = AdaptiveGradientScaler(action_dim)
            layers.append(nn.Tanh())
            self.mu = nn.Sequential(*layers)


# ── Custom SAC Actor with Adaptive Scaling ───────────────────────────────────


class AdaptiveScalingSACActor(sac_policies.Actor):
    """SAC actor with adaptive gradient scaling on mean actions.

    Replaces the Hardtanh clip_mean with AdaptiveGradientScaler, which
    inherently bounds outputs to [-target_range, +target_range].
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        action_dim = get_action_dim(self.action_space)
        self.adaptive_scaler = AdaptiveGradientScaler(action_dim)

        # Replace the Hardtanh in self.mu (if using SDE) with identity —
        # the adaptive scaler will handle bounding
        if self.use_sde and isinstance(self.mu, nn.Sequential):
            # self.mu is Sequential(Linear, Hardtanh) — keep only Linear
            layers = list(self.mu.children())
            layers = [l for l in layers if not isinstance(l, nn.Hardtanh)]
            self.mu = nn.Sequential(*layers) if len(layers) > 1 else layers[0]

    def get_action_dist_params(self, obs):
        features = self.extract_features(obs, self.features_extractor)
        latent_pi = self.latent_pi(features)
        mean_actions = self.mu(latent_pi)

        # Apply adaptive scaling instead of Hardtanh clip
        mean_actions = self.adaptive_scaler(mean_actions)

        if self.use_sde:
            return mean_actions, self.log_std, dict(latent_sde=latent_pi)
        log_std = self.log_std(latent_pi)
        log_std = th.clamp(log_std, sac_policies.LOG_STD_MIN, sac_policies.LOG_STD_MAX)
        return mean_actions, log_std, {}


# ── Adaptive Scaling Policy Classes ──────────────────────────────────────────


class AdaptiveScalingSACPolicy(SACPolicy):
    """SAC policy with adaptive gradient scaling actor, default critic."""

    def make_actor(self, features_extractor=None):
        actor_kwargs = self._update_features_extractor(
            self.actor_kwargs, features_extractor
        )
        return AdaptiveScalingSACActor(**actor_kwargs).to(self.device)


class AdaptiveScalingTD3Policy(TD3Policy):
    """TD3/DDPG policy with adaptive gradient scaling actor, default critic."""

    def make_actor(self, features_extractor=None):
        actor_kwargs = self._update_features_extractor(
            self.actor_kwargs, features_extractor
        )
        return AdaptiveScalingTD3Actor(**actor_kwargs).to(self.device)


class AdaptiveScalingRWAISACPolicy(SACPolicy):
    """SAC policy with adaptive scaling actor + scaled RWAI v2 critic."""

    def __init__(self, *args, q_min: float = 0.0, q_max: float = 500.0, **kwargs):
        self._q_min = q_min
        self._q_max = q_max
        super().__init__(*args, **kwargs)

    def make_actor(self, features_extractor=None):
        actor_kwargs = self._update_features_extractor(
            self.actor_kwargs, features_extractor
        )
        return AdaptiveScalingSACActor(**actor_kwargs).to(self.device)

    def make_critic(self, features_extractor=None):
        critic_kwargs = self._update_features_extractor(
            self.critic_kwargs, features_extractor
        )
        return ScaledRewardAwareContinuousCritic(
            q_min=self._q_min,
            q_max=self._q_max,
            **critic_kwargs,
        ).to(self.device)


class AdaptiveScalingRWAITD3Policy(TD3Policy):
    """TD3/DDPG policy with adaptive scaling actor + scaled RWAI v2 critic."""

    def __init__(self, *args, q_min: float = 0.0, q_max: float = 500.0, **kwargs):
        self._q_min = q_min
        self._q_max = q_max
        super().__init__(*args, **kwargs)

    def make_actor(self, features_extractor=None):
        actor_kwargs = self._update_features_extractor(
            self.actor_kwargs, features_extractor
        )
        return AdaptiveScalingTD3Actor(**actor_kwargs).to(self.device)

    def make_critic(self, features_extractor=None):
        critic_kwargs = self._update_features_extractor(
            self.critic_kwargs, features_extractor
        )
        return ScaledRewardAwareContinuousCritic(
            q_min=self._q_min,
            q_max=self._q_max,
            **critic_kwargs,
        ).to(self.device)

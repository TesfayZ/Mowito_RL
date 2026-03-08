"""Hyperparameter configurations for training experiments."""

import numpy as np
import torch.nn as nn
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise

from custom_policies import (
    RewardAwareSACPolicy,
    RewardAwareTD3Policy,
    ScaledRewardAwareSACPolicy,
    ScaledRewardAwareTD3Policy,
    AdaptiveScalingSACPolicy,
    AdaptiveScalingTD3Policy,
    AdaptiveScalingRWAISACPolicy,
    AdaptiveScalingRWAITD3Policy,
    compute_q_range,
)

# Exploration noise for deterministic off-policy algorithms (TD3, DDPG)
# SAC uses entropy-based exploration so it doesn't need this.
#
# Ornstein-Uhlenbeck noise produces temporally correlated exploration,
# which is critical for swing-up tasks that require sustained momentum.
# Gaussian noise (previous: sigma=0.1) cancels out over consecutive steps,
# preventing the energy-buildup needed for swing-up.
# See: Hollenstein et al. (2022), Eberhard et al. (ICLR 2023).
#
# Parameters: theta=0.15 (mean-reversion rate), sigma=0.3 (volatility)
ACTION_NOISE_SINGLE = OrnsteinUhlenbeckActionNoise(
    mean=np.zeros(1), sigma=0.3 * np.ones(1), theta=0.15
)
ACTION_NOISE_DOUBLE = OrnsteinUhlenbeckActionNoise(
    mean=np.zeros(1), sigma=0.3 * np.ones(1), theta=0.15
)

# ── Q-Value Range Computation ────────────────────────────────────────────────
# Computed via geometric series for dense rewards:
#   Q = R * (1 - gamma^H) / (1 - gamma)
#
# Environment reward bounds (from reward functions):
#   Single (H=500): R ∈ [-0.5, 1], dense reward every step
#     r_min accounts for cart_penalty (0.01*x^2), control_penalty (0.001*a^2),
#     and velocity_penalty near upright. Conservative lower bound: -0.5.
#   Double (H=1000): R ∈ [-0.5, 1], dense reward every step
#     Same structure with slightly different penalty coefficients.

GAMMA = 0.99

Q_MIN_SINGLE, Q_MAX_SINGLE = compute_q_range(
    r_min=-0.5, r_max=1.0, gamma=GAMMA, horizon=500, reward_type="dense"
)
Q_MIN_DOUBLE, Q_MAX_DOUBLE = compute_q_range(
    r_min=-0.5, r_max=1.0, gamma=GAMMA, horizon=1000, reward_type="dense"
)

# ── PPO Configuration ─────────────────────────────────────────────────────────
# On-policy algorithm — less affected by critic initialization bias since
# the value function is re-estimated every rollout. Kept as a baseline.

PPO_SINGLE = dict(
    policy="MlpPolicy",
    learning_rate=3e-4,
    n_steps=1024,
    batch_size=64,
    n_epochs=5,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,
    vf_coef=0.5,
    max_grad_norm=0.5,
    use_sde=True,
    sde_sample_freq=4,
    policy_kwargs=dict(
        net_arch=dict(pi=[64, 64], vf=[64, 64]),
        activation_fn=nn.Tanh,
    ),
    verbose=1,
)

PPO_DOUBLE = dict(
    policy="MlpPolicy",
    learning_rate=3e-4,
    n_steps=1024,
    batch_size=64,
    n_epochs=5,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,
    vf_coef=0.5,
    max_grad_norm=0.5,
    use_sde=True,
    sde_sample_freq=4,
    policy_kwargs=dict(
        net_arch=dict(pi=[64, 64], vf=[64, 64]),
        activation_fn=nn.Tanh,
    ),
    verbose=1,
)

# ── SAC Configuration ─────────────────────────────────────────────────────────
# Off-policy, maximum-entropy. Uses 2 Q-networks (clipped double-Q).
# CPU-optimized: train_freq=32, gradient_steps=1.

SAC_SINGLE = dict(
    policy="MlpPolicy",
    learning_rate=7.3e-4,
    buffer_size=50_000,
    batch_size=128,
    gamma=0.99,
    tau=0.02,
    train_freq=32,
    gradient_steps=1,
    learning_starts=2_000,
    ent_coef="auto",
    use_sde=True,
    policy_kwargs=dict(
        net_arch=[64, 64],
        log_std_init=-3,
    ),
    verbose=1,
)

SAC_DOUBLE = dict(
    policy="MlpPolicy",
    learning_rate=7.3e-4,
    buffer_size=50_000,
    batch_size=128,
    gamma=0.99,
    tau=0.02,
    train_freq=32,
    gradient_steps=1,
    learning_starts=2_000,
    ent_coef="auto",
    use_sde=True,
    policy_kwargs=dict(
        net_arch=[64, 64],
        log_std_init=-3,
    ),
    verbose=1,
)

# ── TD3 Configuration ────────────────────────────────────────────────────────
# Off-policy, deterministic. Uses 2 Q-networks (clipped double-Q) + delayed
# policy updates + target policy smoothing. No entropy term.

TD3_SINGLE = dict(
    policy="MlpPolicy",
    learning_rate=7.3e-4,
    buffer_size=50_000,
    batch_size=128,
    gamma=0.99,
    tau=0.02,
    train_freq=32,
    gradient_steps=1,
    learning_starts=2_000,
    action_noise=ACTION_NOISE_SINGLE,
    policy_kwargs=dict(
        net_arch=[64, 64],
    ),
    verbose=1,
)

TD3_DOUBLE = dict(
    policy="MlpPolicy",
    learning_rate=7.3e-4,
    buffer_size=50_000,
    batch_size=128,
    gamma=0.99,
    tau=0.02,
    train_freq=32,
    gradient_steps=1,
    learning_starts=2_000,
    action_noise=ACTION_NOISE_DOUBLE,
    policy_kwargs=dict(
        net_arch=[64, 64],
    ),
    verbose=1,
)

# ── DDPG Configuration ───────────────────────────────────────────────────────
# Off-policy, deterministic. Single Q-network (no clipped double-Q).
# Most susceptible to overestimation bias — the canonical example.

DDPG_SINGLE = dict(
    policy="MlpPolicy",
    learning_rate=7.3e-4,
    buffer_size=50_000,
    batch_size=128,
    gamma=0.99,
    tau=0.02,
    train_freq=32,
    gradient_steps=1,
    learning_starts=2_000,
    policy_delay=1,
    target_noise_clip=0.0,
    action_noise=ACTION_NOISE_SINGLE,
    policy_kwargs=dict(
        net_arch=[64, 64],
        n_critics=1,
    ),
    verbose=1,
)

DDPG_DOUBLE = dict(
    policy="MlpPolicy",
    learning_rate=7.3e-4,
    buffer_size=50_000,
    batch_size=128,
    gamma=0.99,
    tau=0.02,
    train_freq=32,
    gradient_steps=1,
    learning_starts=2_000,
    policy_delay=1,
    target_noise_clip=0.0,
    action_noise=ACTION_NOISE_DOUBLE,
    policy_kwargs=dict(
        net_arch=[64, 64],
        n_critics=1,
    ),
    verbose=1,
)

# ── Reward-Range-Aware Configurations ────────────────────────────────────────
# Same hyperparameters but with RWAI: last-layer bias shifted to Q_mid.
# Weights are left at SB3 defaults to preserve gradient flow.
# Q-value ranges (Q_MIN_*, Q_MAX_*) computed above from geometric series.
#
# Note: VecNormalize rescales rewards at runtime, so the raw Q-range
# is an approximation. The bias will be overwritten during training
# regardless — the goal is a better starting point than Q ≈ 0.

SAC_SINGLE_RWINIT = dict(
    policy=RewardAwareSACPolicy,
    learning_rate=7.3e-4,
    buffer_size=50_000,
    batch_size=128,
    gamma=0.99,
    tau=0.02,
    train_freq=32,
    gradient_steps=1,
    learning_starts=2_000,
    ent_coef="auto",
    use_sde=True,
    policy_kwargs=dict(
        net_arch=[64, 64],
        log_std_init=-3,
        q_min=Q_MIN_SINGLE,
        q_max=Q_MAX_SINGLE,
    ),
    verbose=1,
)

SAC_DOUBLE_RWINIT = dict(
    policy=RewardAwareSACPolicy,
    learning_rate=7.3e-4,
    buffer_size=50_000,
    batch_size=128,
    gamma=0.99,
    tau=0.02,
    train_freq=32,
    gradient_steps=1,
    learning_starts=2_000,
    ent_coef="auto",
    use_sde=True,
    policy_kwargs=dict(
        net_arch=[64, 64],
        log_std_init=-3,
        q_min=Q_MIN_DOUBLE,
        q_max=Q_MAX_DOUBLE,
    ),
    verbose=1,
)

TD3_SINGLE_RWINIT = dict(
    policy=RewardAwareTD3Policy,
    learning_rate=7.3e-4,
    buffer_size=50_000,
    batch_size=128,
    gamma=0.99,
    tau=0.02,
    train_freq=32,
    gradient_steps=1,
    learning_starts=2_000,
    action_noise=ACTION_NOISE_SINGLE,
    policy_kwargs=dict(
        net_arch=[64, 64],
        q_min=Q_MIN_SINGLE,
        q_max=Q_MAX_SINGLE,
    ),
    verbose=1,
)

TD3_DOUBLE_RWINIT = dict(
    policy=RewardAwareTD3Policy,
    learning_rate=7.3e-4,
    buffer_size=50_000,
    batch_size=128,
    gamma=0.99,
    tau=0.02,
    train_freq=32,
    gradient_steps=1,
    learning_starts=2_000,
    action_noise=ACTION_NOISE_DOUBLE,
    policy_kwargs=dict(
        net_arch=[64, 64],
        q_min=Q_MIN_DOUBLE,
        q_max=Q_MAX_DOUBLE,
    ),
    verbose=1,
)

DDPG_SINGLE_RWINIT = dict(
    policy=RewardAwareTD3Policy,  # DDPG uses TD3 policy in SB3
    learning_rate=7.3e-4,
    buffer_size=50_000,
    batch_size=128,
    gamma=0.99,
    tau=0.02,
    train_freq=32,
    gradient_steps=1,
    learning_starts=2_000,
    policy_delay=1,
    target_noise_clip=0.0,
    action_noise=ACTION_NOISE_SINGLE,
    policy_kwargs=dict(
        net_arch=[64, 64],
        n_critics=1,
        q_min=Q_MIN_SINGLE,
        q_max=Q_MAX_SINGLE,
    ),
    verbose=1,
)

DDPG_DOUBLE_RWINIT = dict(
    policy=RewardAwareTD3Policy,  # DDPG uses TD3 policy in SB3
    learning_rate=7.3e-4,
    buffer_size=50_000,
    batch_size=128,
    gamma=0.99,
    tau=0.02,
    train_freq=32,
    gradient_steps=1,
    learning_starts=2_000,
    policy_delay=1,
    target_noise_clip=0.0,
    action_noise=ACTION_NOISE_DOUBLE,
    policy_kwargs=dict(
        net_arch=[64, 64],
        n_critics=1,
        q_min=Q_MIN_DOUBLE,
        q_max=Q_MAX_DOUBLE,
    ),
    verbose=1,
)

# ── Scaled RWAI (v2) Configurations ─────────────────────────────────────────
# Scaled RWAI: last-layer weights empirically calibrated so output std spans
# the Q-value range. Different states get meaningfully different initial
# Q-estimates instead of all concentrating around Q_mid.
#
# CRITICAL: These use norm_reward=False to avoid the VecNormalize confound.
# With norm_reward=True, the raw Q-range [0, ~100] overshoots the normalized
# training targets, causing the same collapse seen in v1 RWAI.

SAC_SINGLE_RWINIT_V2 = dict(
    policy=ScaledRewardAwareSACPolicy,
    learning_rate=7.3e-4,
    buffer_size=50_000,
    batch_size=128,
    gamma=0.99,
    tau=0.02,
    train_freq=32,
    gradient_steps=1,
    learning_starts=2_000,
    ent_coef="auto",
    use_sde=True,
    policy_kwargs=dict(
        net_arch=[64, 64],
        log_std_init=-3,
        q_min=Q_MIN_SINGLE,
        q_max=Q_MAX_SINGLE,
    ),
    verbose=1,
)

SAC_DOUBLE_RWINIT_V2 = dict(
    policy=ScaledRewardAwareSACPolicy,
    learning_rate=7.3e-4,
    buffer_size=50_000,
    batch_size=128,
    gamma=0.99,
    tau=0.02,
    train_freq=32,
    gradient_steps=1,
    learning_starts=2_000,
    ent_coef="auto",
    use_sde=True,
    policy_kwargs=dict(
        net_arch=[64, 64],
        log_std_init=-3,
        q_min=Q_MIN_DOUBLE,
        q_max=Q_MAX_DOUBLE,
    ),
    verbose=1,
)

TD3_SINGLE_RWINIT_V2 = dict(
    policy=ScaledRewardAwareTD3Policy,
    learning_rate=7.3e-4,
    buffer_size=50_000,
    batch_size=128,
    gamma=0.99,
    tau=0.02,
    train_freq=32,
    gradient_steps=1,
    learning_starts=2_000,
    action_noise=ACTION_NOISE_SINGLE,
    policy_kwargs=dict(
        net_arch=[64, 64],
        q_min=Q_MIN_SINGLE,
        q_max=Q_MAX_SINGLE,
    ),
    verbose=1,
)

TD3_DOUBLE_RWINIT_V2 = dict(
    policy=ScaledRewardAwareTD3Policy,
    learning_rate=7.3e-4,
    buffer_size=50_000,
    batch_size=128,
    gamma=0.99,
    tau=0.02,
    train_freq=32,
    gradient_steps=1,
    learning_starts=2_000,
    action_noise=ACTION_NOISE_DOUBLE,
    policy_kwargs=dict(
        net_arch=[64, 64],
        q_min=Q_MIN_DOUBLE,
        q_max=Q_MAX_DOUBLE,
    ),
    verbose=1,
)

DDPG_SINGLE_RWINIT_V2 = dict(
    policy=ScaledRewardAwareTD3Policy,
    learning_rate=7.3e-4,
    buffer_size=50_000,
    batch_size=128,
    gamma=0.99,
    tau=0.02,
    train_freq=32,
    gradient_steps=1,
    learning_starts=2_000,
    policy_delay=1,
    target_noise_clip=0.0,
    action_noise=ACTION_NOISE_SINGLE,
    policy_kwargs=dict(
        net_arch=[64, 64],
        n_critics=1,
        q_min=Q_MIN_SINGLE,
        q_max=Q_MAX_SINGLE,
    ),
    verbose=1,
)

DDPG_DOUBLE_RWINIT_V2 = dict(
    policy=ScaledRewardAwareTD3Policy,
    learning_rate=7.3e-4,
    buffer_size=50_000,
    batch_size=128,
    gamma=0.99,
    tau=0.02,
    train_freq=32,
    gradient_steps=1,
    learning_starts=2_000,
    policy_delay=1,
    target_noise_clip=0.0,
    action_noise=ACTION_NOISE_DOUBLE,
    policy_kwargs=dict(
        net_arch=[64, 64],
        n_critics=1,
        q_min=Q_MIN_DOUBLE,
        q_max=Q_MAX_DOUBLE,
    ),
    verbose=1,
)

# ── Adaptive Gradient Scaling Configurations (for GCAS experiments) ───────────────────
# Same hyperparameters as baseline but with adaptive gradient scaling actors.
# Used with gradient-clipped algorithm classes from gc_algorithms.py.

SAC_SINGLE_AS = dict(
    policy=AdaptiveScalingSACPolicy,
    learning_rate=7.3e-4,
    buffer_size=50_000,
    batch_size=128,
    gamma=0.99,
    tau=0.02,
    train_freq=32,
    gradient_steps=1,
    learning_starts=2_000,
    ent_coef="auto",
    use_sde=True,
    policy_kwargs=dict(
        net_arch=[64, 64],
        log_std_init=-3,
    ),
    verbose=1,
)

SAC_DOUBLE_AS = dict(
    policy=AdaptiveScalingSACPolicy,
    learning_rate=7.3e-4,
    buffer_size=50_000,
    batch_size=128,
    gamma=0.99,
    tau=0.02,
    train_freq=32,
    gradient_steps=1,
    learning_starts=2_000,
    ent_coef="auto",
    use_sde=True,
    policy_kwargs=dict(
        net_arch=[64, 64],
        log_std_init=-3,
    ),
    verbose=1,
)

TD3_SINGLE_AS = dict(
    policy=AdaptiveScalingTD3Policy,
    learning_rate=7.3e-4,
    buffer_size=50_000,
    batch_size=128,
    gamma=0.99,
    tau=0.02,
    train_freq=32,
    gradient_steps=1,
    learning_starts=2_000,
    action_noise=ACTION_NOISE_SINGLE,
    policy_kwargs=dict(
        net_arch=[64, 64],
    ),
    verbose=1,
)

TD3_DOUBLE_AS = dict(
    policy=AdaptiveScalingTD3Policy,
    learning_rate=7.3e-4,
    buffer_size=50_000,
    batch_size=128,
    gamma=0.99,
    tau=0.02,
    train_freq=32,
    gradient_steps=1,
    learning_starts=2_000,
    action_noise=ACTION_NOISE_DOUBLE,
    policy_kwargs=dict(
        net_arch=[64, 64],
    ),
    verbose=1,
)

DDPG_SINGLE_AS = dict(
    policy=AdaptiveScalingTD3Policy,
    learning_rate=7.3e-4,
    buffer_size=50_000,
    batch_size=128,
    gamma=0.99,
    tau=0.02,
    train_freq=32,
    gradient_steps=1,
    learning_starts=2_000,
    policy_delay=1,
    target_noise_clip=0.0,
    action_noise=ACTION_NOISE_SINGLE,
    policy_kwargs=dict(
        net_arch=[64, 64],
        n_critics=1,
    ),
    verbose=1,
)

DDPG_DOUBLE_AS = dict(
    policy=AdaptiveScalingTD3Policy,
    learning_rate=7.3e-4,
    buffer_size=50_000,
    batch_size=128,
    gamma=0.99,
    tau=0.02,
    train_freq=32,
    gradient_steps=1,
    learning_starts=2_000,
    policy_delay=1,
    target_noise_clip=0.0,
    action_noise=ACTION_NOISE_DOUBLE,
    policy_kwargs=dict(
        net_arch=[64, 64],
        n_critics=1,
    ),
    verbose=1,
)

# ── RWAI v2 + Adaptive Gradient Scaling Configurations ───────────────────────────────

SAC_SINGLE_RWINIT_V2_AS = dict(
    policy=AdaptiveScalingRWAISACPolicy,
    learning_rate=7.3e-4,
    buffer_size=50_000,
    batch_size=128,
    gamma=0.99,
    tau=0.02,
    train_freq=32,
    gradient_steps=1,
    learning_starts=2_000,
    ent_coef="auto",
    use_sde=True,
    policy_kwargs=dict(
        net_arch=[64, 64],
        log_std_init=-3,
        q_min=Q_MIN_SINGLE,
        q_max=Q_MAX_SINGLE,
    ),
    verbose=1,
)

SAC_DOUBLE_RWINIT_V2_AS = dict(
    policy=AdaptiveScalingRWAISACPolicy,
    learning_rate=7.3e-4,
    buffer_size=50_000,
    batch_size=128,
    gamma=0.99,
    tau=0.02,
    train_freq=32,
    gradient_steps=1,
    learning_starts=2_000,
    ent_coef="auto",
    use_sde=True,
    policy_kwargs=dict(
        net_arch=[64, 64],
        log_std_init=-3,
        q_min=Q_MIN_DOUBLE,
        q_max=Q_MAX_DOUBLE,
    ),
    verbose=1,
)

TD3_SINGLE_RWINIT_V2_AS = dict(
    policy=AdaptiveScalingRWAITD3Policy,
    learning_rate=7.3e-4,
    buffer_size=50_000,
    batch_size=128,
    gamma=0.99,
    tau=0.02,
    train_freq=32,
    gradient_steps=1,
    learning_starts=2_000,
    action_noise=ACTION_NOISE_SINGLE,
    policy_kwargs=dict(
        net_arch=[64, 64],
        q_min=Q_MIN_SINGLE,
        q_max=Q_MAX_SINGLE,
    ),
    verbose=1,
)

TD3_DOUBLE_RWINIT_V2_AS = dict(
    policy=AdaptiveScalingRWAITD3Policy,
    learning_rate=7.3e-4,
    buffer_size=50_000,
    batch_size=128,
    gamma=0.99,
    tau=0.02,
    train_freq=32,
    gradient_steps=1,
    learning_starts=2_000,
    action_noise=ACTION_NOISE_DOUBLE,
    policy_kwargs=dict(
        net_arch=[64, 64],
        q_min=Q_MIN_DOUBLE,
        q_max=Q_MAX_DOUBLE,
    ),
    verbose=1,
)

DDPG_SINGLE_RWINIT_V2_AS = dict(
    policy=AdaptiveScalingRWAITD3Policy,
    learning_rate=7.3e-4,
    buffer_size=50_000,
    batch_size=128,
    gamma=0.99,
    tau=0.02,
    train_freq=32,
    gradient_steps=1,
    learning_starts=2_000,
    policy_delay=1,
    target_noise_clip=0.0,
    action_noise=ACTION_NOISE_SINGLE,
    policy_kwargs=dict(
        net_arch=[64, 64],
        n_critics=1,
        q_min=Q_MIN_SINGLE,
        q_max=Q_MAX_SINGLE,
    ),
    verbose=1,
)

DDPG_DOUBLE_RWINIT_V2_AS = dict(
    policy=AdaptiveScalingRWAITD3Policy,
    learning_rate=7.3e-4,
    buffer_size=50_000,
    batch_size=128,
    gamma=0.99,
    tau=0.02,
    train_freq=32,
    gradient_steps=1,
    learning_starts=2_000,
    policy_delay=1,
    target_noise_clip=0.0,
    action_noise=ACTION_NOISE_DOUBLE,
    policy_kwargs=dict(
        net_arch=[64, 64],
        n_critics=1,
        q_min=Q_MIN_DOUBLE,
        q_max=Q_MAX_DOUBLE,
    ),
    verbose=1,
)

# ── Training Configuration ────────────────────────────────────────────────────
# All experiments use the same timesteps and seed for fair comparison.

TOTAL_TIMESTEPS = 500_000
SEED = 42

# ── PER Hyperparameters ──────────────────────────────────────────────────────
# From Schaul et al. (2015), standard values for proportional PER.
PER_ALPHA = 0.6        # Priority exponent (0 = uniform, 1 = full prioritization)
PER_BETA_INIT = 0.4    # Initial IS weight exponent
PER_BETA_FINAL = 1.0   # Final IS weight exponent (annealed linearly)

# ── Experiment Registry ───────────────────────────────────────────────────────
# Generated programmatically for complete combinatorial coverage.
#
# All off-policy experiments use gradient clipping (max_grad_norm=1.0).
# The four contributions ablated combinatorially are:
#   RWAI v2   — Reward-range-aware critic init (scaled weights)
#   PER       — Prioritized Experience Replay
#   AS        — Adaptive Gradient Scaling (prevents tanh saturation)
#   QBound    — Q-value bounding to theoretical range
#
# This gives 2^4 = 16 variants × 3 algos × 2 envs = 96 off-policy experiments,
# plus 2 PPO baselines, 6 legacy RWAI v1, 6 vanilla baselines (for v1 comparison),
# and 6 norm_reward=False baselines = 116 total.

EXPERIMENTS = {}

# ── PPO baselines (on-policy, no contributions applicable) ──
EXPERIMENTS["ppo_single"] = {
    "env_id": "CartPoleSwingUp-v0",
    "algo": "PPO",
    "config": PPO_SINGLE,
    "timesteps": TOTAL_TIMESTEPS,
    "n_envs": 4,
    "seed": SEED,
}
EXPERIMENTS["ppo_double"] = {
    "env_id": "DoubleCartPoleSwingUp-v0",
    "algo": "PPO",
    "config": PPO_DOUBLE,
    "timesteps": TOTAL_TIMESTEPS,
    "n_envs": 4,
    "seed": SEED,
}
# ── Off-policy experiments (programmatic generation) ─────────────────────────
# All off-policy experiments use GC (gradient clipping, max_grad_norm=1.0).
# The four contributions ablated combinatorially: RWAI v2, PER, AS, QBound.

_ALGOS = ["sac", "td3", "ddpg"]
_ENVS = {
    "single": {
        "env_id": "CartPoleSwingUp-v0",
        "qbound_min": Q_MIN_SINGLE,
        "qbound_max": Q_MAX_SINGLE,
    },
    "double": {
        "env_id": "DoubleCartPoleSwingUp-v0",
        "qbound_min": Q_MIN_DOUBLE,
        "qbound_max": Q_MAX_DOUBLE,
    },
}

# Config dict lookup: (algo, env) -> base config
# Indexed by (algo, env, rwai, as) -> config dict
_CONFIG_MAP = {
    ("sac", "single", False, False): SAC_SINGLE,
    ("sac", "double", False, False): SAC_DOUBLE,
    ("td3", "single", False, False): TD3_SINGLE,
    ("td3", "double", False, False): TD3_DOUBLE,
    ("ddpg", "single", False, False): DDPG_SINGLE,
    ("ddpg", "double", False, False): DDPG_DOUBLE,
    ("sac", "single", True, False): SAC_SINGLE_RWINIT_V2,
    ("sac", "double", True, False): SAC_DOUBLE_RWINIT_V2,
    ("td3", "single", True, False): TD3_SINGLE_RWINIT_V2,
    ("td3", "double", True, False): TD3_DOUBLE_RWINIT_V2,
    ("ddpg", "single", True, False): DDPG_SINGLE_RWINIT_V2,
    ("ddpg", "double", True, False): DDPG_DOUBLE_RWINIT_V2,
    ("sac", "single", False, True): SAC_SINGLE_AS,
    ("sac", "double", False, True): SAC_DOUBLE_AS,
    ("td3", "single", False, True): TD3_SINGLE_AS,
    ("td3", "double", False, True): TD3_DOUBLE_AS,
    ("ddpg", "single", False, True): DDPG_SINGLE_AS,
    ("ddpg", "double", False, True): DDPG_DOUBLE_AS,
    ("sac", "single", True, True): SAC_SINGLE_RWINIT_V2_AS,
    ("sac", "double", True, True): SAC_DOUBLE_RWINIT_V2_AS,
    ("td3", "single", True, True): TD3_SINGLE_RWINIT_V2_AS,
    ("td3", "double", True, True): TD3_DOUBLE_RWINIT_V2_AS,
    ("ddpg", "single", True, True): DDPG_SINGLE_RWINIT_V2_AS,
    ("ddpg", "double", True, True): DDPG_DOUBLE_RWINIT_V2_AS,
}

# Algo class lookup: (per, qbound) -> algo class suffix
# GC is always included.
_ALGO_CLASS_MAP = {
    (False, False): "_GC",       # e.g. SAC_GC
    (True, False): "_PER_GC",    # e.g. SAC_PER_GC
    (False, True): "_GC_QB",     # e.g. SAC_GC_QB
    (True, True): "_PER_GC_QB",  # e.g. SAC_PER_GC_QB
}

import itertools

for algo in _ALGOS:
    for env_name, env_info in _ENVS.items():
        # Generate all 2^4 = 16 combinations of {RWAI v2, PER, AS, QBound}
        for rwai, per, adap_scale, qbound in itertools.product([False, True], repeat=4):
            # Build experiment name
            parts = [algo, env_name]
            if rwai:
                parts.append("rwinit_v2")
            if per:
                parts.append("per")
            if adap_scale:
                parts.append("as")
            if qbound:
                parts.append("qbound")
            exp_name = "_".join(parts)

            # Select config dict (determined by rwai + as)
            config = _CONFIG_MAP[(algo, env_name, rwai, adap_scale)]

            # Select algo class (determined by per + qbound; GC always on)
            algo_class = algo.upper() + _ALGO_CLASS_MAP[(per, qbound)]

            # Build experiment entry
            entry = {
                "env_id": env_info["env_id"],
                "algo": algo_class,
                "config": config,
                "timesteps": TOTAL_TIMESTEPS,
                "n_envs": 1,
                "seed": SEED,
                "max_grad_norm": 1.0,
            }

            # RWAI v2 and QBound require raw (unnormalized) rewards
            if rwai or qbound:
                entry["norm_reward"] = False

            # PER config
            if per:
                entry["per"] = {
                    "alpha": PER_ALPHA,
                    "beta_init": PER_BETA_INIT,
                    "beta_final": PER_BETA_FINAL,
                }

            # QBound config
            if qbound:
                entry["qbound"] = {
                    "qbound_min": env_info["qbound_min"],
                    "qbound_max": env_info["qbound_max"],
                }

            EXPERIMENTS[exp_name] = entry

# ── norm_reward=False baselines ───────────────────────────────────────────────
# Required for fair comparison with RWAI v2 and QBound experiments, which
# disable reward normalization. Without these, any performance difference
# could be attributed to norm_reward rather than the technique itself.

for algo in _ALGOS:
    for env_name, env_info in _ENVS.items():
        _base_config = _CONFIG_MAP[(algo, env_name, False, False)]
        _algo_class = algo.upper() + "_GC"
        exp_name = f"{algo}_{env_name}_no_norm_reward"
        EXPERIMENTS[exp_name] = {
            "env_id": env_info["env_id"],
            "algo": _algo_class,
            "config": _base_config,
            "timesteps": TOTAL_TIMESTEPS,
            "n_envs": 1,
            "seed": SEED,
            "norm_reward": False,
            "max_grad_norm": 1.0,
        }

# ── Vanilla baselines (no GC, norm_reward=True) for RWAI v1 comparison ───────
# These reproduce the original baseline conditions (before GC was added) so the
# v1 legacy table has a fair "Default" comparison column.
for algo in _ALGOS:
    for env_name, env_info in _ENVS.items():
        _base_config = _CONFIG_MAP[(algo, env_name, False, False)]
        exp_name = f"{algo}_{env_name}_vanilla"
        EXPERIMENTS[exp_name] = {
            "env_id": env_info["env_id"],
            "algo": algo.upper(),
            "config": _base_config,
            "timesteps": TOTAL_TIMESTEPS,
            "n_envs": 1,
            "seed": SEED,
        }

# ── Legacy RWAI v1 (bias-only, retained for diagnostic reference) ────────────
# These use vanilla algo classes (no GC) and norm_reward=True (default) to
# reproduce the original VecNormalize confound: critic bias set to Q_mid ≈ 25
# while VecNormalize rescales rewards to ≈[0, 10], causing overestimation.
for algo in _ALGOS:
    for env_name, env_info in _ENVS.items():
        _rwinit_config = {
            ("sac", "single"): SAC_SINGLE_RWINIT,
            ("sac", "double"): SAC_DOUBLE_RWINIT,
            ("td3", "single"): TD3_SINGLE_RWINIT,
            ("td3", "double"): TD3_DOUBLE_RWINIT,
            ("ddpg", "single"): DDPG_SINGLE_RWINIT,
            ("ddpg", "double"): DDPG_DOUBLE_RWINIT,
        }
        EXPERIMENTS[f"{algo}_{env_name}_rwinit"] = {
            "env_id": env_info["env_id"],
            "algo": algo.upper(),
            "config": _rwinit_config[(algo, env_name)],
            "timesteps": TOTAL_TIMESTEPS,
            "n_envs": 1,
            "seed": SEED,
        }

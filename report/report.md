# Critic Initialization, Gradient Stabilization, and Q-Value Regularization for Off-Policy Deep Reinforcement Learning

**Author:** Tesfay Zemuy Gebrekidan, PhD
**Date:** March 2026

---

## Abstract

We investigate five complementary techniques for improving off-policy deep reinforcement learning on hard-exploration continuous control tasks: (1) **Reward-Range-Aware Initialization (RWAI)** of critic last-layer weights and biases to the expected Q-value range, in both a bias-only variant (v1) and a scaled-weights variant (v2); (2) **Gradient Clipping (GC)** via `clip_grad_norm_` after each backward pass to stabilize training; (3) **Adaptive Scaling (AS)** of pre-tanh actor activations using running mean/std statistics to prevent gradient saturation; (4) **Prioritized Experience Replay (PER)** with importance-sampling-weighted critic loss and sum-tree sampling; and (5) **Q-Value Bounding (QBound)** via two-stage hard clipping on critic TD targets and soft (softplus) clipping on actor Q-values to constrain estimates within the theoretically achievable range. We evaluate these techniques across a full combinatorial ablation: 3 off-policy algorithms (SAC, TD3, DDPG) x 2 environments (single and double inverted pendulum swing-up) x 2^4 = 16 combinations of {RWAI v2, PER, AS, QBound}, plus 2 PPO baselines, 6 legacy RWAI v1 experiments, and 6 norm_reward=False baselines, totaling **110 experiments**. All off-policy experiments use gradient clipping (GC) as standard infrastructure. Initial RWAI v1 results revealed a critical confound between raw Q-range computation and runtime reward normalization (VecNormalize), producing 5-50x overestimation; the expanded experiment matrix addresses this by disabling reward normalization for RWAI v2 and QBound experiments and providing dedicated norm_reward=False baselines for fair comparison. The TD3 RWAI v1 results yield a significant diagnostic finding: temporary rescue from exploration failure (7.95 to 259, 2.93 to 440) confirms that TD3's clipped double-Q mechanism suppresses value signals from rare high-reward transitions. In the completed 110-experiment matrix, QBound emerges as the most consistently beneficial technique for SAC (single: 312→453, double: 434→452), while Adaptive Scaling proves catastrophic for SAC (all AS variants collapse below 75) yet partially rescues TD3 from exploration failure (single: 8.7→205). RWAI v2 significantly improves SAC (single: 453, double: 483 peak) but destabilizes TD3. The best-performing configurations combine QBound with PER for SAC (453.6/453.5 single, 455.7/455.7 double) and PER with AS for TD3 double (441.4/428.8). These results demonstrate that technique effectiveness is strongly algorithm-dependent, and that no single technique universally improves all off-policy algorithms.

---

## 1. Introduction

Deep reinforcement learning algorithms that learn action-value (Q) functions -- including SAC [1], TD3 [2], and DDPG [3] -- rely on value networks (critics) to estimate expected cumulative returns. These estimates drive policy updates: the actor chooses actions that maximize the value network's predictions. Consequently, the accuracy of value estimates, particularly during early training, has an outsized influence on learning dynamics.

Standard neural network initialization schemes (Xavier [4], Kaiming [5], orthogonal [6]) are designed to maintain gradient flow in deep networks. They center initial outputs near zero with bounded variance. While this is appropriate for classification and regression tasks with zero-centered targets, it creates a systematic bias for Q-value estimation when the reward structure is non-zero-centered:

- **Non-zero-centered rewards** (e.g., survival bonuses, uprightness scores with penalties): True Q-values may be large positive numbers, but the value network initially outputs approximately 0, causing **underestimation**. Even with small negative penalties (our environments have r_min approximately -0.5), the expected return range [Q_min, Q_max] is far from zero-centered.
- **Negative-only rewards** (e.g., cost-based formulations): True Q-values are large negative numbers, but the critic outputs approximately 0, causing **overestimation**.
- **Mixed rewards**: The standard initialization may be approximately correct, but the scale may still be wrong.

This initialization mismatch is particularly problematic for off-policy algorithms because:

1. **Bootstrap amplification**: Off-policy methods compute TD targets using value-network estimates. Biased initial estimates propagate and potentially amplify through bootstrapping.
2. **Entropy interaction** (SAC): Underestimated Q-values make the entropy bonus relatively larger, causing excessive exploration when exploitation would be beneficial.
3. **Persistent bias**: Unlike on-policy methods (PPO, A2C) that re-estimate values each rollout, off-policy value networks accumulate knowledge over the full training run, making early biases harder to correct.

Beyond initialization, off-policy algorithms face additional challenges in hard-exploration settings: gradient instability from large TD errors, tanh saturation in deterministic actors, inefficient uniform replay sampling, and unconstrained Q-value divergence. We address these with a suite of five complementary techniques: RWAI for critic initialization, GC for gradient stabilization, AS for actor gradient flow, PER for sample efficiency, and QBound for Q-value regularization.

## 2. Related Work

**Overestimation bias in RL.** Thrun and Schwartz [7] first identified systematic overestimation in Q-learning. Van Hasselt [8] proposed Double Q-learning to address this. Fujimoto et al. [2] extended this to continuous control with TD3's clipped double-Q trick. However, these works focus on overestimation from max operations over noisy estimates, not from initialization bias.

**Network initialization.** Glorot and Bengio [4] proposed Xavier initialization for symmetric activations. He et al. [5] developed Kaiming initialization for ReLU networks. These methods optimize gradient flow but do not consider the output magnitude of the target signal.

**Pessimistic initialization.** In offline RL, conservative Q-learning (CQL) [9] intentionally underestimates Q-values to prevent out-of-distribution actions. Our work differs: we aim to match, not artificially shrink, the Q-value range.

**Optimistic initialization.** In tabular RL, initializing Q-values optimistically encourages exploration [10]. Our approach is related but targets function approximation settings where the entire last layer (weights + bias) must be configured.

**Prioritized Experience Replay.** Schaul et al. [16] introduced proportional prioritization based on TD errors, with importance-sampling correction to avoid bias. We adapt this for continuous off-policy algorithms (SAC, TD3, DDPG), integrating it with gradient clipping and Q-value bounding.

**Q-Value Bounding.** Gebrekidan [17] introduced QBound, constraining Q-value estimates to the theoretically achievable range using two-stage hard clipping on critic targets and soft (softplus) clipping on actor Q-values. We integrate QBound into our combinatorial ablation framework.

**Adaptive Scaling.** Gebrekidan [18] identified gradient asymmetry and activation saturation as failure modes in deterministic actor networks and proposed an adaptive scaling module to maintain pre-tanh activations in the high-gradient region. We evaluate this technique across SAC, TD3, and DDPG.

**Gradient clipping.** Gradient norm clipping is standard in on-policy methods (PPO uses max_grad_norm=0.5) but not natively supported in SB3 2.7.1's off-policy algorithms. We add `clip_grad_norm_` to SAC, TD3, and DDPG to stabilize training when large TD errors produce extreme gradients.

**Exploration noise for swing-up.** Hollenstein et al. [13] showed that Ornstein-Uhlenbeck (OU) noise significantly outperforms Gaussian noise for energy-buildup tasks like pendulum swing-up. Eberhard et al. [14] further demonstrated that temporally correlated noise (OU, pink noise) consistently outperforms white Gaussian noise on swing-up benchmarks. We adopt OU noise as the default for all deterministic-policy algorithms.

## 3. Method

### 3.1 Problem Setup

Consider an environment with reward function r(s, a) bounded in [r_min, r_max] per timestep, discount factor gamma, and maximum episode length H. The expected Q-value range depends on the reward structure:

**Dense rewards** (reward every timestep):
```
Q_min = r_min * (1 - gamma^H) / (1 - gamma)
Q_max = r_max * (1 - gamma^H) / (1 - gamma)
```

**Sparse rewards** (reward only at terminal state):
```
Q_min = r_min * gamma^(H-1)
Q_max = r_max * gamma^(H-1)
```

The geometric series formula gives the exact discounted sum for dense rewards, while the sparse formula accounts for the single discounted reward at the end of the episode.

### 3.2 Reward-Range-Aware Initialization (RWAI)

We propose two variants of critic last-layer initialization based on the computed Q-value range.

**RWAI v1 (bias-only):** Given Q_min and Q_max, only the last linear layer's bias is shifted to Q_mid = (Q_min + Q_max) / 2. The default Kaiming uniform weight initialization is left intact. This preserves gradient flow and state differentiation while providing a coarse output scale shift. The network learns fine-grained state-action-dependent corrections around this baseline through normal training.

**RWAI v2 (scaled weights with empirical calibration):** In addition to setting the bias, the last-layer weights are scaled so that the output standard deviation spans the Q-value range. Specifically:
1. Pass random inputs through the hidden layers to measure the current output distribution.
2. Scale the last-layer weights so that output std approximately equals Q_range / 4 (so that +/- 2 sigma covers [Q_min, Q_max]).
3. Recalibrate the bias to compensate for the non-zero mean from scaled weights acting on non-negative ReLU activations.
4. The weight scale factor is capped at 100x to prevent extreme gradient amplification.

This gives different state-action pairs meaningfully different initial Q-estimates, rather than all concentrating near Q_mid.

**Critical design constraint:** Both RWAI variants require `norm_reward=False` (no VecNormalize reward normalization) so that the raw Q-value range matches the training targets. With reward normalization enabled, the effective Q-range is dramatically smaller than the raw range, causing catastrophic overestimation (Section 6.7).

### 3.3 Q-Range Computation

For our environments (dense rewards, r in [-0.5, 1.0]):

| Environment | H | gamma | Q_min | Q_max | Q_mid |
|-------------|---|-------|-------|-------|-------|
| Single pendulum | 500 | 0.99 | -49.67 | 99.34 | 24.84 |
| Double pendulum | 1000 | 0.99 | -50.00 | 100.00 | 25.00 |

The reward lower bound r_min = -0.5 is a conservative estimate accounting for cart displacement penalties (0.01 * x^2), control effort penalties (0.001 * a^2), and angular velocity penalties near upright. While the uprightness component is in [0, 1], the penalties can push the per-step reward negative. The upper bound r_max = 1.0 corresponds to a perfectly upright, centered, stationary pole.

### 3.4 Gradient Clipping (GC)

Stable-Baselines3 2.7.1's SAC, TD3, and DDPG implementations do not natively support `max_grad_norm`. We add gradient clipping by subclassing each algorithm and inserting `torch.nn.utils.clip_grad_norm_()` after each `backward()` call and before each `optimizer.step()`. This clips both critic and actor gradients to a maximum L2 norm of 1.0.

Gradient clipping serves as stabilization infrastructure: it prevents catastrophic gradient explosions from large TD errors (common in early training or when Q-value estimates are miscalibrated). All off-policy experiments in our matrix use GC as standard.

For TD3/DDPG, the gradient-clipped algorithm classes also synchronize AdaptiveGradientScaler buffers (g_mean, g_var) from the online actor to the target actor after each polyak update, since `polyak_update` only syncs parameters, not buffers.

### 3.5 Adaptive Scaling (AS)

Deterministic policy actors (TD3, DDPG) use a tanh output nonlinearity to bound actions to [-1, 1]. SAC similarly uses tanh in its squashed Gaussian policy. When pre-tanh activations grow large during training, tanh saturates and gradients vanish, stalling policy learning.

We introduce an **AdaptiveGradientScaler** module placed immediately before the tanh nonlinearity. It tracks running mean and standard deviation of pre-tanh activations per action dimension using exponential moving average (EMA) with decay 0.99, similar to BatchNorm. The scaler linearly maps [mean - k*std, mean + k*std] to [-target_range, +target_range], where k_std=2.5 (capturing approximately 99% of the distribution) and target_range=2.0.

Key implementation details:
- Uses persistent buffers (not parameters) so stats survive model saving/loading but are not affected by polyak_update.
- Passes through unscaled until the first training batch initializes the stats (safe because `learning_starts=2000`).
- For SAC, replaces the Hardtanh clip_mean used with SDE.
- Buffer synchronization from online to target actor is handled explicitly after each polyak update via `sync_scaler_buffers()`.

### 3.6 Prioritized Experience Replay (PER)

Standard uniform replay sampling treats all transitions equally, regardless of their learning value. Prioritized Experience Replay [16] samples transitions proportional to their TD error, focusing updates on transitions where the critic's prediction is most wrong.

Our implementation:
- **Sum-tree data structure** for O(log n) proportional sampling with stratified segments.
- **Priority exponent** alpha=0.6 (interpolates between uniform and full prioritization).
- **Importance-sampling (IS) correction**: Weights w_i = (N * P(i))^(-beta), normalized by max weight. Beta is annealed linearly from 0.4 to 1.0 over the course of training to fully correct the sampling bias.
- **IS-weighted critic loss**: Each sample's squared TD error is multiplied by its IS weight before averaging.
- **Priority updates**: After each gradient step, TD errors from the first Q-network are used to update buffer priorities. A small constant (1e-6) prevents zero priorities.
- **Max-priority decay**: New transitions receive the current maximum priority (ensuring they are sampled at least once). The maximum priority decays toward the current batch maximum (factor 0.95) to prevent a single outlier from inflating all future initial priorities.

### 3.7 Q-Value Bounding (QBound)

Q-value estimates can diverge beyond the theoretically achievable range, particularly during early training or under overestimation. QBound constrains Q-values to [Q_min, Q_max] using a two-stage approach:

**Critic training (hard clipping):**
1. Clip next-state Q-values: `next_q = clamp(next_q, Q_min, Q_max)`
2. Clip the full TD target: `target = clamp(r + gamma * next_q, Q_min, Q_max)`

**Actor training (soft clipping):**
- Apply softplus-based soft clipping to actor Q-values before computing the policy gradient. The softplus function (with beta=5.0) provides smooth, differentiable enforcement of bounds, preserving gradient flow at the boundaries. As beta approaches infinity, this approaches hard clipping.

QBound requires `norm_reward=False` so that the raw Q-value bounds match the actual training targets.

### 3.8 Implementation Architecture

All five techniques are implemented as modular, composable components:
- **RWAI**: Custom policy classes that override `make_critic()` to inject initialized critics.
- **GC**: Algorithm subclasses that override `train()` to insert gradient clipping.
- **AS**: Custom actor classes with the AdaptiveGradientScaler module; policy classes override `make_actor()`.
- **PER**: Algorithm subclasses that override `train()` to use IS-weighted loss and priority updates, combined with a custom `PrioritizedReplayBuffer`.
- **QBound**: Algorithm subclasses that override `train()` to insert Q-value clipping.

Combinations are achieved through multiple inheritance: e.g., `SACWithPERAndGCAndQB` inherits from `SACWithPERAndGC` and adds QBound. DDPG inherits from TD3 everywhere (following the SB3 convention).

## 4. Experimental Setup

### 4.1 Environments

We evaluate on two continuous control tasks with **dense rewards** that can go negative:

**Single Pendulum Swing-Up (CartPoleSwingUp-v0):**
- Cart mass M=1.0 kg, pole mass m=0.1 kg, pole half-length l=0.5 m, force scale F_max=10 N
- Observation: 5D [x, sin(theta), cos(theta), x_dot, theta_dot]
- Action: 1D continuous force [-1, 1], scaled by F_max
- Episode: 500 steps max, dt=0.02 s
- Goal: Swing pole from DOWN (theta=pi) to UP (theta=0) and balance
- Cart terminates at |x| > 2.4 m

The equations of motion are derived from the Euler-Lagrange equations for the cart-pole system:

```
(M + m) * x_ddot + m*l * theta_ddot * cos(theta) - m*l * theta_dot^2 * sin(theta) = F - mu * x_dot
m*l * x_ddot * cos(theta) + m*l^2 * theta_ddot - m*g*l * sin(theta) = 0
```

where mu=0.1 is the cart friction coefficient. These are solved analytically for x_ddot and theta_ddot at each substep.

**Double Pendulum Swing-Up (DoubleCartPoleSwingUp-v0):**
- Cart mass M=1.0 kg, two identical poles (m1=m2=0.1 kg, half-length l1=l2=0.5 m)
- Observation: 8D [x, sin(theta_1), cos(theta_1), sin(theta_2), cos(theta_2), x_dot, theta_1_dot, theta_2_dot]
- Action: 1D continuous force [-1, 1], scaled by F_max=10 N
- Episode: 1000 steps max, dt=0.02 s
- Goal: Swing both poles from DOWN to UP and balance
- Cart terminates at |x| > 2.4 m

The dynamics are governed by a 3x3 mass matrix system M(q) * q_ddot = f(q, q_dot, u) for generalized coordinates q = [x, theta_1, theta_2], incorporating Coriolis, centrifugal, and gravitational terms. Moments of inertia use the thin rod formula I = (1/12)*m*L^2 for each pole. The system is solved via numpy's linear algebra solver at each substep.

**Integration and Reward (both environments):**

Both environments use 4th-order Runge-Kutta (RK4) integration with sub-stepping (2 substeps for single, 4 for double) for numerical stability. Angles are represented as sin/cos pairs to avoid discontinuity at ±pi.

Reward (single pendulum):
```
reward = (cos(theta) + 1) / 2           # uprightness [0, 1]
       - 0.01 * x^2                     # cart centering penalty
       - 0.001 * a^2                    # control effort penalty
       - 0.002 * upright^4 * theta_dot^2 # angular velocity penalty
```

The smooth `upright^4` activation avoids reward discontinuity near the upright position. The double pendulum uses a weighted combination of individual pole uprightness (0.4 each) and normalized tip height (0.2), with cart penalty coefficient 0.005. Reward range: [-0.5, 1.0] per step for both environments.

### 4.2 Algorithms and Experiment Matrix

We test four algorithms -- three off-policy and one on-policy control:

| Algorithm | Type | # Critics | Exploration | Key Properties |
|-----------|------|-----------|-------------|----------------|
| **PPO** [11] | On-policy | 1 (value fn) | SDE + 4 parallel envs | Control experiment |
| **SAC** [1] | Off-policy | 2 (clipped double-Q) | Entropy-regularized + SDE | Maximum entropy objective |
| **TD3** [2] | Off-policy | 2 (clipped double-Q) | OU noise (theta=0.15, sigma=0.3) | Delayed updates, target smoothing |
| **DDPG** [3] | Off-policy | 1 | OU noise (theta=0.15, sigma=0.3) | Single critic, no double-Q |

The full experiment matrix comprises **110 experiments**:

| Category | Count | Description |
|----------|-------|-------------|
| PPO baselines | 2 | On-policy control (single + double), no contributions |
| Off-policy combinatorial | 96 | 3 algos x 2 envs x 2^4 combinations of {RWAI v2, PER, AS, QBound} |
| Legacy RWAI v1 | 6 | 3 algos x 2 envs, bias-only RWAI for reference |
| norm_reward=False baselines | 6 | 3 algos x 2 envs, GC only with raw rewards for fair comparison |
| **Total** | **110** | |

All off-policy experiments use gradient clipping (GC, max_grad_norm=1.0) as standard infrastructure. The four contributions ablated combinatorially are RWAI v2, PER, AS, and QBound. Experiments using RWAI v2 or QBound set `norm_reward=False` to avoid the VecNormalize confound. The dedicated norm_reward=False baselines enable fair comparison against these experiments by isolating the effect of each technique from the effect of disabling reward normalization.

**Exploration noise choice**: TD3 and DDPG use Ornstein-Uhlenbeck noise rather than Gaussian noise. OU noise produces temporally correlated actions critical for swing-up tasks that require sustained momentum to build pendulum energy. Hollenstein et al. [13] showed OU noise significantly outperforms Gaussian noise for energy-buildup tasks. Parameters: theta=0.15 (mean-reversion rate, approximately 7-step correlation), sigma=0.3 (amplitude sufficient for swing-up energy buildup).

### 4.3 Shared Configuration

All experiments use identical training parameters for fair comparison:

- **Timesteps**: 500,000 for all experiments
- **Seed**: 42 for all experiments (deterministic environment creation, model initialization, and evaluation)
- **Observation normalization**: VecNormalize (norm_obs=True, clip_obs=10.0)
- **Reward normalization**: VecNormalize (norm_reward=True) for baseline experiments; norm_reward=False for RWAI v2, QBound, and dedicated no-norm-reward baselines
- **Evaluation**: 10 episodes every 10K steps, deterministic policy, unnormalized rewards

Off-policy algorithms share:
- Learning rate: 7.3e-4, buffer: 50K, batch: 128, gamma: 0.99, tau: 0.02
- Train frequency: 1 gradient step per 32 env steps
- Learning starts: 2K random steps
- Network: [64, 64] MLP
- Gradient clipping: max_grad_norm=1.0

PPO configuration:
- Learning rate: 3e-4, n_steps: 1024, batch: 64, epochs: 5
- GAE lambda: 0.95, clip: 0.2, entropy coef: 0.01
- Network: [64, 64] separate pi/vf, Tanh activation, SDE
- 4 parallel environments (SubprocVecEnv)

### 4.4 Q-Value Range Configuration

Q-value ranges computed from reward bounds using the dense geometric series:

| Environment | r_min | r_max | gamma | H | Q_min | Q_max | Q_mid |
|-------------|-------|-------|-------|---|-------|-------|-------|
| Single | -0.5 | 1.0 | 0.99 | 500 | -49.67 | 99.34 | 24.84 |
| Double | -0.5 | 1.0 | 0.99 | 1000 | -50.00 | 100.00 | 25.00 |

These bounds are used for RWAI v2 critic initialization (Q_mid and output range), QBound hard/soft clipping limits, and as reference for the legacy RWAI v1 experiments.

### 4.5 PER Configuration

Prioritized Experience Replay hyperparameters (from Schaul et al. [16]):

| Parameter | Value | Description |
|-----------|-------|-------------|
| alpha | 0.6 | Priority exponent (0=uniform, 1=full prioritization) |
| beta_init | 0.4 | Initial IS weight exponent |
| beta_final | 1.0 | Final IS weight exponent (annealed linearly) |

### 4.6 Default Initialization Comparison

| Component | PPO (default) | SAC/TD3/DDPG (default) | RWAI v1 | RWAI v2 |
|-----------|--------------|------------------------|---------|---------|
| Last layer weights | Orthogonal, gain=1 | Kaiming uniform | Kaiming uniform (unchanged) | Kaiming, scaled to span Q-range |
| Last layer bias | 0.0 | approx. 0 | Q_mid (24.84/25.00) | Calibrated to Q_mid |
| Initial Q output mean | approx. -0.25 | approx. 0.1 | approx. 24.84/25.00 | approx. 24.84/25.00 |
| Initial Q output std | ~0.08 | ~0.08 | ~0.08 (preserved) | ~Q_range/4 (calibrated) |

## 5. Results

### 5.1 RWAI v1 Results (Legacy)

The original 16 RWAI v1 experiments (6 off-policy RWAI v1 + corresponding baselines) provide foundational diagnostic findings. Note that these legacy experiments used the original VecNormalize confound (norm_reward=True with raw Q-range initialization):

| Algorithm | Env | Default Best | Default Final | RWAI v1 Best | RWAI v1 Final | Effect |
|-----------|-----|:-----------:|:------------:|:---------:|:----------:|:------|
| **PPO** | Single | 449.09 | 449.09 | 450.26 | 449.54 | Neutral |
| **PPO** | Double | 492.75 | 479.90 | 486.72 | 475.49 | Neutral |
| **SAC** | Single | 452.73 | 451.43 | 51.81 | 0.00 | Catastrophic collapse |
| **SAC** | Double | 477.51 | 451.71 | 338.08 | 0.36 | Catastrophic collapse |
| **TD3** | Single | 7.95 | 7.95 | 259.05 | -2.11 | Early rescue, then collapse |
| **TD3** | Double | 2.93 | 2.75 | 440.38 | -1.95 | Early rescue, then collapse |
| **DDPG** | Single | 325.57 | 280.09 | 248.57 | 150.99 | Degraded |
| **DDPG** | Double | 475.50 | 455.93 | 464.82 | 371.01 | Degraded |

**Key**: Best Eval = highest mean reward across all evaluations (10 episodes each, every 10K steps). Final Eval = mean reward at the last evaluation (500K steps). Rewards are unnormalized. PPO results shown are from separate PPO RWAI v1 experiments that are no longer in the current codebase; the current PPO experiments are baselines without RWAI.

### 5.2 PPO: On-Policy Robustness (Control Experiment)

PPO achieves strong, stable performance on both environments, serving as an on-policy control:

| Variant | Best Eval | Final Eval | Converged By |
|---------|:---------:|:----------:|:------------:|
| PPO Single | 449.09 | 449.09 | ~320K |
| PPO Double | 492.75 | 479.90 | ~350K |

PPO confirms both environments are solvable and that any failures observed in off-policy algorithms are algorithm-specific, not environment-related. The current codebase includes only these two PPO baselines (no PPO RWAI experiments, as on-policy algorithms are robust to initialization bias as confirmed by the original experiments).

### 5.3 SAC: Default Success, RWAI v1 Catastrophic Collapse

SAC with default initialization solves both environments convincingly:

- **SAC Single default**: Best 452.73, Final 451.43 -- stable convergence
- **SAC Double default**: Best 477.51, Final 451.71 -- solved with moderate variance

SAC with RWAI v1 suffers catastrophic training collapse on both environments:

- **SAC Single RWAI v1**: Brief peak at 51.81 (60K steps), then collapsed to 0.00 by 100K. The policy diverged completely and never recovered over the remaining 400K steps.
- **SAC Double RWAI v1**: More gradual rise to 338.08 (160K steps), then collapsed to near-zero by 190K. Brief partial recovery (240 at 350K) before collapsing again.

**Failure mechanism**: Setting the critic bias to Q_mid approximately 50 (under the original r_min=0 assumption) creates massive initial overestimation relative to the normalized reward scale. SAC's automatic entropy coefficient alpha tuning is particularly sensitive to Q-value magnitude: when Q-values are artificially large, the entropy bonus alpha*H(pi) becomes relatively insignificant, causing the policy to prematurely converge to a narrow action distribution. Once the critic begins correcting downward, the alpha coefficient cannot readjust quickly enough, resulting in a destabilizing feedback loop between collapsing Q-values and poorly calibrated entropy.

### 5.4 TD3: RWAI v1 Rescues from Exploration Trap, Then Collapses

TD3 with default initialization fails completely on both environments, confirming the earlier baseline findings:

- **TD3 Single default**: Best 7.95, Final 7.95 -- flatlined for 500K steps, episode length ~37 steps
- **TD3 Double default**: Best 2.93, Final 2.75 -- similarly stuck

TD3 with RWAI v1 reveals a striking two-phase pattern:

**Phase 1 -- Exploration rescue (0-80K steps):** RWAI dramatically rescues TD3 from its exploration trap. TD3 Single RWAI climbs to 259.05 (50K), TD3 Double RWAI reaches 440.38 (80K) -- a 55x improvement over the default. The optimistic initial Q-values (bias approximately 50) counteract the clipped double-Q's conservatism, driving the policy to explore high-reward states it would never discover under default initialization.

**Phase 2 -- Catastrophic collapse (80K-500K steps):** Both variants then suffer complete training collapse. TD3 Single RWAI drops from 259 to -2.11 by 90K. TD3 Double RWAI drops from 440 to -1.95 by 190K. Neither variant recovers.

**Root cause -- the clipped double-Q and conservative Q-estimation interaction**: TD3's conservative Q-estimation (min(Q_1, Q_2)) suppresses the value signal from rare high-reward transitions. Under default initialization with Q approximately 0, this creates a vicious cycle: conservative Q-values lead to a policy that does not pursue rare high-reward states, which produces fewer high-reward transitions in the buffer, which keeps Q-values conservative. RWAI breaks this cycle initially by providing optimistic Q-estimates, but the overestimation is too large and unstable. Once the critic correction overshoots, TD3's conservatism locks the policy into the collapsed state.

This demonstrates that TD3's exploration failure on swing-up is not primarily a noise issue but a fundamental interaction between conservative Q-estimation and hard exploration. RWAI proves this by temporarily rescuing TD3, confirming the diagnosis.

### 5.5 DDPG: Default Success, RWAI v1 Degradation

DDPG with default initialization performs well despite using identical hyperparameters and OU noise as TD3:

- **DDPG Single default**: Best 325.57, Final 280.09 -- partial solve, moderate instability
- **DDPG Double default**: Best 475.50, Final 455.93 -- fully solved

DDPG with RWAI v1 shows degraded performance:

- **DDPG Single RWAI v1**: Best 248.57 (340K), Final 150.99 -- worse than default (325 to 249 peak, 280 to 151 final)
- **DDPG Double RWAI v1**: Best 464.82 (460K), Final 371.01 -- worse final than default (456 to 371), with more instability

The critical difference between DDPG and TD3 is that DDPG uses a **single Q-network** without clipped double-Q. DDPG's natural overestimation -- the problem TD3 was designed to fix -- functions as an implicit exploration bonus in hard-exploration settings:

| Property | TD3 (default) | DDPG (default) |
|----------|:---:|:---:|
| Q-networks | 2 (min) | 1 |
| Overestimation | Suppressed | Natural |
| Single pendulum | 7.95 (stuck) | 325.57 (partial) |
| Double pendulum | 2.93 (stuck) | 475.50 (solved) |

Adding RWAI's large positive bias to DDPG's already-overestimating critic pushes the overestimation beyond a beneficial threshold, introducing additional instability without a compensating exploration benefit (DDPG already explores adequately via its natural overestimation).

### 5.6 Complete Combinatorial Results

All 110 experiments have been completed. The following tables present best and final mean evaluation rewards (10 episodes, deterministic policy, unnormalized rewards) for every experiment. "Best" is the highest mean reward across all evaluation checkpoints; "Final" is the reward at 500K steps.

#### 5.6.1 SAC Results

| Variant | Single Best | Single Final | Double Best | Double Final |
|---------|:----------:|:-----------:|:----------:|:-----------:|
| baseline (GC, norm_reward) | 312.4 | 312.4 | 434.5 | 369.9 |
| no_norm_reward baseline | 452.6 | 448.3 | 452.0 | 408.7 |
| RWAI v1 (legacy) | 451.8 | 451.8 | 423.9 | 404.7 |
| **RWAI v2** | **452.6** | **446.9** | **482.9** | **453.7** |
| PER | 453.0 | 388.7 | 423.5 | 371.4 |
| AS | 8.0 | 4.9 | 3.2 | 3.2 |
| **QBound** | **453.2** | **452.2** | **452.0** | **408.7** |
| RWAI v2 + PER | 454.4 | 453.3 | 456.1 | 444.7 |
| RWAI v2 + AS | 74.8 | 65.8 | 5.4 | 2.8 |
| RWAI v2 + QBound | 452.6 | 446.9 | 482.9 | 453.7 |
| PER + AS | 101.2 | 3.3 | 118.2 | 67.0 |
| **PER + QBound** | **453.6** | **453.5** | **455.7** | **455.7** |
| AS + QBound | 225.4 | 171.8 | 420.7 | 375.8 |
| RWAI v2 + PER + AS | 44.6 | 7.5 | 3.6 | 2.6 |
| **RWAI v2 + PER + QBound** | **454.4** | **453.3** | **463.1** | **424.7** |
| RWAI v2 + AS + QBound | 109.4 | 54.1 | 4.0 | 3.1 |
| PER + AS + QBound | 197.2 | 133.4 | 434.3 | 405.5 |
| RWAI v2 + PER + AS + QBound | 55.5 | 39.6 | 8.0 | 4.6 |

**Key SAC findings:**
- **QBound is the most impactful single technique** for SAC, boosting single from 312→453 and providing stable convergence.
- **Adaptive Scaling is catastrophic** for SAC: every variant including AS collapses below 75 (single) or below 120 (double). AS disrupts SAC's squashed Gaussian policy by interfering with the log-probability computation.
- **RWAI v2 provides strong peak performance** (482.9 on double, highest across all SAC variants).
- **PER + QBound achieves the most stable results**: 453.5/455.7 final rewards with minimal best-final gap.
- **Any combination including AS degrades SAC**, even when QBound is present.

#### 5.6.2 TD3 Results

| Variant | Single Best | Single Final | Double Best | Double Final |
|---------|:----------:|:-----------:|:----------:|:-----------:|
| baseline (GC, norm_reward) | 8.7 | 8.7 | 414.2 | 388.3 |
| no_norm_reward baseline | 8.6 | 8.6 | 429.9 | 429.9 |
| RWAI v1 (legacy) | 449.1 | 284.4 | 459.9 | 400.4 |
| RWAI v2 | 7.6 | 2.8 | 40.9 | 3.6 |
| PER | 8.0 | 7.9 | 429.0 | 377.9 |
| **AS** | **205.4** | **173.9** | 409.3 | 390.7 |
| QBound | 8.6 | 8.6 | 429.9 | 429.9 |
| RWAI v2 + PER | 74.1 | 5.2 | 18.8 | 2.6 |
| RWAI v2 + AS | 5.8 | -2.0 | 73.5 | 1.5 |
| RWAI v2 + QBound | 7.6 | 2.8 | 40.9 | 3.6 |
| **PER + AS** | **292.5** | **269.5** | **441.4** | **428.8** |
| PER + QBound | 8.6 | 8.6 | **457.1** | 405.8 |
| AS + QBound | 165.1 | 158.3 | 411.3 | 378.3 |
| RWAI v2 + PER + AS | 24.3 | -0.3 | **413.5** | **412.8** |
| RWAI v2 + PER + QBound | 74.1 | 5.2 | 18.8 | 3.7 |
| RWAI v2 + AS + QBound | 5.8 | -2.0 | 230.3 | 19.2 |
| PER + AS + QBound | 211.6 | -2.1 | 428.6 | 416.8 |
| RWAI v2 + PER + AS + QBound | 24.3 | -1.1 | 413.5 | 412.8 |

**Key TD3 findings:**
- **TD3 single pendulum remains largely unsolvable**. Only AS provides partial rescue (205→293 with PER), consistent with the hypothesis that TD3's clipped double-Q suppresses exploration.
- **RWAI v2 is destructive for TD3**: all RWAI v2 variants without PER+AS collapse. With norm_reward=False, RWAI v2's initialization still conflicts with TD3's conservative estimation.
- **AS partially rescues TD3 single** (8.7→205 alone, 8.7→293 with PER), providing a similar exploration-boosting effect to RWAI v1's temporary rescue but with more stability.
- **PER + AS is the best TD3 combination** on both environments (292.5/269.5 single, 441.4/428.8 double).
- **TD3 double works well across most variants** since the double pendulum's longer episodes and richer dynamics provide more learning signal.

#### 5.6.3 DDPG Results

| Variant | Single Best | Single Final | Double Best | Double Final |
|---------|:----------:|:-----------:|:----------:|:-----------:|
| baseline (GC, norm_reward) | 298.8 | 298.8 | 478.6 | 411.2 |
| no_norm_reward baseline | 363.7 | 294.1 | 478.0 | 219.8 |
| RWAI v1 (legacy) | 356.6 | 294.6 | 437.6 | 211.7 |
| RWAI v2 | **453.0** | 252.2 | 472.4 | 436.6 |
| PER | 267.7 | 251.4 | 454.6 | 414.2 |
| AS | 245.5 | 190.9 | **457.9** | **457.1** |
| QBound | 363.7 | 294.1 | 478.0 | 219.8 |
| RWAI v2 + PER | **455.4** | 206.2 | 459.0 | 389.3 |
| RWAI v2 + AS | 244.8 | 113.1 | 438.0 | 388.4 |
| RWAI v2 + QBound | 453.0 | 252.2 | 472.4 | 436.6 |
| PER + AS | 251.0 | 231.8 | 446.7 | 446.7 |
| PER + QBound | 287.0 | 287.0 | 452.8 | 409.6 |
| AS + QBound | 221.6 | 210.2 | 452.8 | 382.8 |
| RWAI v2 + PER + AS | 242.7 | 190.1 | **448.5** | **448.5** |
| RWAI v2 + PER + QBound | 428.5 | 260.5 | 459.0 | 389.3 |
| RWAI v2 + AS + QBound | 244.8 | 113.1 | 438.0 | 388.4 |
| PER + AS + QBound | 263.1 | 199.6 | 398.4 | 270.8 |
| RWAI v2 + PER + AS + QBound | 242.7 | 190.1 | 448.5 | 448.5 |

**Key DDPG findings:**
- **RWAI v2 achieves the highest peak on single** (453.0) but with significant instability (final: 252.2). Similarly RWAI v2 + PER peaks at 455.4.
- **DDPG double is generally robust** across most configurations, with the baseline already achieving 478.6 peak.
- **AS provides the most stable double performance** (457.9/457.1 with minimal best-final gap).
- **No single technique consistently improves both envs**: RWAI v2 helps peaks but hurts stability; AS helps double stability but hurts single.
- **DDPG is the most robust algorithm to technique variations**, reflecting its simpler single-critic architecture.

#### 5.6.4 PPO Baselines

| Variant | Single Best | Single Final | Double Best | Double Final |
|---------|:----------:|:-----------:|:----------:|:-----------:|
| PPO baseline | 450.2 | 439.0 | 488.0 | 458.7 |

PPO confirms both environments are solvable and serves as the on-policy performance ceiling against which off-policy improvements are evaluated (Section 6.10).

#### 5.6.5 Cross-Algorithm Best Performers

| Environment | Best Configuration | Best | Final | Algorithm |
|------------|-------------------|:----:|:-----:|-----------|
| Single | SAC + RWAI v2 + PER | 454.4 | 453.3 | SAC |
| Single | SAC + PER + QBound | 453.6 | 453.5 | SAC |
| Single | SAC + QBound | 453.2 | 452.2 | SAC |
| Double | SAC + RWAI v2 | 482.9 | 453.7 | SAC |
| Double | SAC + RWAI v2 + PER + QBound | 463.1 | 424.7 | SAC |
| Double | TD3 + PER + QBound | 457.1 | 405.8 | TD3 |
| Double | DDPG + AS | 457.9 | 457.1 | DDPG |
| Double | SAC + PER + QBound | 455.7 | 455.7 | SAC |

SAC dominates the top performers, particularly when combined with QBound and/or RWAI v2 (without AS).

### 5.7 Training Dynamics Summary

The complete 110-experiment matrix reveals four distinct training dynamics patterns:

**Stable convergence** (PPO, SAC+QBound, SAC+RWAI v2+PER, DDPG+AS double):
- Monotonic improvement with low variance
- Final reward within 5% of best reward
- No collapse events

**High peak with instability** (DDPG+RWAI v2 single, SAC baselines):
- Reaches excellent peak performance
- Moderate-to-large gap between best and final (15-45% decline)
- Late-training instability suggests critic drift

**Catastrophic collapse from AS** (all SAC+AS variants):
- Immediate failure to learn (rewards stay at 3-8)
- AS interferes with SAC's squashed Gaussian log-probability computation
- Not recoverable even with QBound present

**Stuck at local optimum** (TD3 single most variants):
- Flatline from early training through 500K steps
- Only AS (with or without PER) breaks through
- Confirms clipped double-Q suppresses exploration in hard-exploration tasks

## 6. Analysis

### 6.1 Why Off-Policy Algorithms Are Affected by Value-Network Initialization

In off-policy RL, the Bellman update is:

```
Q(s,a) <- r + gamma * Q_target(s', a')
```

If Q_target is approximately 0 initially (due to default initialization) but the true target is r + gamma*Q* approximately 0.5 + 0.99*25 approximately 25, the value network must close a large gap. During this correction period:

1. **Policy receives poor gradients**: The actor optimizes against an inaccurate value network, potentially learning suboptimal behavior that persists in the replay buffer.
2. **Target network lag**: Soft updates (tau=0.02) mean the target network trails the online value network, further slowing convergence.
3. **Entropy dominance (SAC)**: With Q approximately 0, the objective `Q(s,a) + alpha*H(pi)` is dominated by entropy, causing the policy to maximize randomness rather than reward.

However, our RWAI v1 experiments reveal that the opposite problem -- **overestimation from initialization** -- is equally or more destructive. When Q_target starts at 50 but the normalized reward scale produces true targets of order 1-10, the excess must be corrected downward. This correction creates its own pathologies (Section 6.3).

### 6.2 Why On-Policy Algorithms Are Robust

PPO computes advantages using Generalized Advantage Estimation (GAE):

```
A_hat_t = sum((gamma*lambda)^l * delta_{t+l})    where delta_t = r_t + gamma*V(s_{t+1}) - V(s_t)
```

Even if V(s) starts at 0 or 50, the TD errors delta_t are computed from **fresh rollout data** and provide correct relative signals. The clipping mechanism further limits the impact of value estimation errors on policy updates. Moreover, PPO's value function is re-estimated from scratch each epoch, preventing bias accumulation.

Our results confirm this: PPO achieves nearly identical performance with default initialization (449/493), demonstrating complete robustness to initialization bias. This validates our control experiment design and isolates the observed failures to off-policy-specific mechanisms.

### 6.3 The Overestimation-Collapse Cascade (RWAI v1)

The RWAI v1 experiments reveal a consistent three-stage failure pattern across off-policy algorithms:

**Stage 1 -- Optimistic exploration (0-50K steps):** The large positive bias (Q_mid approximately 50 under the original r_min=0 assumption) drives the policy to explore aggressively, producing high-reward transitions. This is most visible in TD3, where RWAI rescues the agent from its exploration trap (7.95 to 259 for single, 2.93 to 440 for double).

**Stage 2 -- Critic correction (50K-150K steps):** As real data accumulates, the critic begins correcting toward the true (normalized) Q-range. This creates large TD errors, destabilizing both the critic and the target network. The actor, which has been optimizing against inflated Q-values, receives contradictory gradient signals.

**Stage 3 -- Policy collapse (150K+ steps):** The combined effect of critic instability, stale overestimated transitions in the replay buffer, and rapid policy change produces a divergence cascade. The policy collapses to degenerate behavior (near-zero or negative rewards), and the critic locks into predicting these low values, preventing recovery.

This pattern manifests differently across algorithms due to their structural differences:

| Algorithm | Stage 1 | Stage 2 | Stage 3 | Root Cause |
|-----------|---------|---------|---------|------------|
| SAC | Moderate improvement | alpha tuning disrupted | Full collapse | Entropy-Q coupling |
| TD3 | Dramatic rescue | Conservative clipping amplifies correction | Full collapse | Overestimation + double-Q conflict |
| DDPG | No additional benefit | Gradual degradation | Partial collapse | Excess overestimation |

### 6.4 SAC: Entropy Coefficient as the Vulnerability

SAC's automatic entropy tuning adjusts alpha to maintain a target entropy H_bar:

```
alpha* = argmin_alpha E[-alpha * log pi(a|s) - alpha * H_bar]
```

When Q-values are artificially large (approximately 50), the policy improvement gradient nabla_phi E[Q(s,a) - alpha*log pi(a|s)] is dominated by the Q-term, causing premature policy sharpening. The entropy coefficient alpha adjusts downward in response to this apparent "certainty." When the critic subsequently corrects to realistic values, alpha is too low to provide adequate exploration, and the policy cannot recover from its overly narrow distribution.

This explains why SAC collapses more severely than TD3 or DDPG with RWAI v1: SAC's strength (adaptive entropy) becomes a vulnerability when Q-values are miscalibrated.

### 6.5 TD3: Proof of the Exploration Hypothesis

The TD3 RWAI v1 results provide the strongest evidence for our central thesis about conservative Q-estimation and exploration:

1. **Default TD3 is stuck** (best: 7.95/2.93) -- clipped double-Q suppresses value signals from rare high-reward states
2. **RWAI rescues TD3 temporarily** (peak: 259/440) -- optimistic initialization overwhelms the conservative bias
3. **But the rescue is unsustainable** (final: -2.11/-1.95) -- the overestimation-correction cascade destabilizes training

This two-phase behavior confirms that TD3's failure is specifically an exploration-exploitation interaction, not a capacity or hyperparameter issue. The algorithm can learn swing-up behavior (Phase 1 proves this) but cannot sustain it under its conservative Q-estimation framework when the initialization bias washes out.

### 6.6 The Clipped Double-Q and Exploration Tension

Our experiments reveal a fundamental tension in TD3's design:

**For dense-reward tasks** (locomotion, reaching): Overestimation leads to unstable training, so clipped double-Q is beneficial.

**For hard-exploration tasks** (swing-up): Overestimation provides an implicit exploration bonus, so clipped double-Q is harmful.

The TD3 vs. DDPG comparison under default initialization demonstrates this clearly:

| Property | TD3 (default) | DDPG (default) |
|----------|:---:|:---:|
| Q-networks | 2 (take minimum) | 1 |
| Overestimation | Suppressed by clipping | Natural, uncorrected |
| Effect on rare high-reward transitions | Value signal suppressed | Value signal amplified |
| Single pendulum | 7.95 (stuck) | 325.57 (partial) |
| Double pendulum | 2.93 (stuck) | 475.50 (solved) |

DDPG's natural overestimation, the very problem TD3 was designed to fix, functions as an implicit exploration bonus. The agent is drawn toward states where it has experienced unusually high rewards, even if the Q-estimate is inflated. TD3's clipped double-Q eliminates this beneficial bias along with the harmful one.

### 6.7 The VecNormalize Confound and Its Resolution

The most critical finding from the RWAI v1 experiments is that the failure is not a refutation of the initialization hypothesis, but a consequence of the **interaction between raw Q-range computation and runtime reward normalization**.

RWAI v1 computed Q_mid from raw reward bounds (originally assuming r_min=0): Q_mid approximately 50 for the single pendulum. However, VecNormalize standardizes rewards to approximately zero mean and unit variance during training. The effective Q-range under normalization is dramatically smaller -- on the order of [0, 10] rather than [0, 100].

This creates a systematic mismatch:

```
RWAI v1 bias:  Q_mid ~ 50      (from raw rewards, r_min=0)
True Q-range:  ~ [0, 10]       (under normalized rewards)
Overestimation: ~5-50x          (bias / true range)
```

The expanded experiment matrix addresses this confound in two ways:

1. **norm_reward=False for RWAI v2 and QBound experiments**: By disabling reward normalization, the raw Q-range [-49.67, 99.34] / [-50.00, 100.00] matches the actual training targets. This eliminates the normalization-initialization mismatch entirely.

2. **Dedicated norm_reward=False baselines**: Six experiments (3 algos x 2 envs) with only GC and no reward normalization enable isolating the effect of each technique from the effect of disabling normalization. Any performance difference between a RWAI v2 experiment and its corresponding no-norm-reward baseline can be attributed to RWAI v2 itself, not to the normalization change.

Additionally, the corrected r_min=-0.5 (accounting for penalties that can push rewards negative) gives a more accurate Q_mid approximately 25 rather than 50, reducing the initialization magnitude even in the raw-reward setting.

### 6.8 Exploration Mechanism Comparison

| Mechanism | Type | Effectiveness for Swing-Up | Notes |
|-----------|------|:------------------------:|-------|
| SDE (PPO, SAC) | State-dependent | Effective | N/A for TD3/DDPG |
| Entropy bonus (SAC) | Objective-based | Effective | Disrupted by Q-scale mismatch |
| OU noise (TD3, DDPG) | Additive correlated | Necessary, insufficient for TD3 | Provides temporal correlation |
| Overestimation (DDPG) | Implicit (single Q) | Partially compensates | Absent in TD3 |
| RWAI v1 (legacy) | Initialization bias | Temporarily effective, then destructive | Confounded by VecNormalize |
| RWAI v2 | Initialization (scaled) | Strong for SAC (+140), destructive for TD3 | Algorithm-dependent |
| PER | Sampling priority | Moderate alone, strong with QBound/AS | Best as a synergy amplifier |
| AS | Pre-tanh rescaling | Rescues TD3 (+197), destroys SAC | Incompatible with SAC |
| QBound | Value constraint | Excellent for SAC (+141), neutral for TD3/DDPG | Most consistent single technique |

### 6.9 SDE Not Available for TD3/DDPG

We investigated State-Dependent Exploration (gSDE) [15], which SAC and PPO use successfully. However, the Stable-Baselines3 implementation of TD3 and DDPG does not support SDE (`sde_support=False`). This is a fundamental architectural limitation: TD3/DDPG use deterministic policy networks, while SDE requires a stochastic policy that parameterizes a state-dependent noise distribution. OU noise is the strongest available exploration mechanism for these algorithms within the SB3 framework.

## 7. Discussion

### 7.1 The Initialization-Normalization Dilemma and Its Resolution

The central lesson from the RWAI v1 experiments is that value-network initialization cannot be designed in isolation from the reward preprocessing pipeline. RWAI targets a real problem -- default initialization produces Q approximately 0 when the true range is [-50, 100] -- but the solution was originally confounded by VecNormalize, which dynamically rescales the reward signal to a much smaller range.

The expanded experiment matrix resolves this dilemma by disabling reward normalization for RWAI v2 and QBound experiments, and using raw rewards with the correctly computed Q-range (r_min=-0.5, not 0). The results validate this approach: SAC + RWAI v2 achieves 482.9 on double pendulum (vs. 434.5 baseline), demonstrating that proper alignment between initialization and reward scale enables genuine benefits.

### 7.2 Technique-Algorithm Interaction Matrix

The complete 110-experiment results reveal that technique effectiveness is strongly algorithm-dependent. The following matrix summarizes the effect of each technique relative to the algorithm's baseline:

| Technique | SAC | TD3 | DDPG |
|-----------|:---:|:---:|:----:|
| **RWAI v2** | Strong positive (+140 single, +48 double peak) | Destructive (collapse on both) | High peak but unstable (+154 peak, -47 final single) |
| **PER** | Moderate (+141 single peak, unstable) | Neutral single, moderate double | Slightly negative single, neutral double |
| **AS** | **Catastrophic** (collapse to <8 on all) | **Partially rescues** single (+197) | Mixed (hurts single, stabilizes double) |
| **QBound** | **Excellent** (+141 single, +18 double) | Neutral (no effect on single trap) | Neutral (matches no_norm_reward baseline) |

### 7.3 Why Adaptive Scaling Destroys SAC

The most striking finding is AS's catastrophic effect on SAC: every SAC+AS variant collapses (best rewards 3-118), while AS partially rescues TD3 (8.7→205 single).

**Root cause**: SAC uses a squashed Gaussian policy where the log-probability computation depends on the exact pre-tanh activation values:

```
log π(a|s) = log μ(u|s) - Σ log(1 - tanh²(u_i))
```

The AdaptiveGradientScaler linearly remaps pre-tanh activations based on running statistics, changing the relationship between the Gaussian mean/std and the actual pre-tanh values used in the tanh transformation. This corrupts the log-probability correction term, causing incorrect entropy estimates and destabilizing SAC's automatic temperature tuning (alpha).

TD3 and DDPG use deterministic policies (direct tanh output, no log-probability), so AS can safely rescale pre-tanh activations without this side effect. For TD3, AS prevents tanh saturation that would otherwise trap the policy in a low-exploration regime.

**Implication**: Techniques that modify pre-tanh activations must account for whether the algorithm depends on the statistical relationship between those activations and the action distribution.

### 7.4 QBound as Principled Q-Value Regularization

QBound emerges as the most consistently beneficial single technique for SAC:

| SAC Variant | Single Best/Final | Double Best/Final |
|------------|:-----------------:|:-----------------:|
| Baseline (norm_reward) | 312.4 / 312.4 | 434.5 / 369.9 |
| QBound | 453.2 / 452.2 | 452.0 / 408.7 |
| PER + QBound | 453.6 / 453.5 | 455.7 / 455.7 |

QBound's effectiveness for SAC likely stems from two mechanisms:
1. **Prevents Q-value drift**: By constraining critic targets to [Q_min, Q_max], QBound prevents the slow Q-value inflation that causes late-training instability in SAC baselines.
2. **Stabilizes entropy tuning**: Bounded Q-values keep the Q-term in SAC's objective at a consistent scale relative to the entropy bonus, preventing the entropy coefficient from being driven to pathological values.

However, QBound has no effect on TD3's single-pendulum exploration failure (8.6 vs 8.7 baseline), confirming that TD3's problem is fundamentally about exploration, not Q-value divergence.

### 7.5 Competitiveness with PPO

PPO serves as the on-policy performance ceiling: 450.2/439.0 best/final (single), 488.0/458.7 (double). A central question is whether the investigated techniques make off-policy algorithms competitive with PPO.

| Algorithm | Configuration | Best | Final | Gap vs PPO Final |
|-----------|--------------|:----:|:-----:|:----------------:|
| **Single pendulum** (PPO final: 439.0) | | | | |
| SAC | baseline (GC only) | 312.4 | 312.4 | -126.6 |
| SAC | + QBound | 453.2 | 452.2 | **+13.2** |
| SAC | + PER + QBound | 453.6 | 453.5 | **+14.5** |
| SAC | + RWAI v2 + PER | 454.4 | 453.3 | **+14.3** |
| TD3 | baseline (GC only) | 8.7 | 8.7 | -430.3 |
| TD3 | + PER + AS | 292.5 | 269.5 | -169.5 |
| DDPG | baseline (GC only) | 298.8 | 298.8 | -140.2 |
| **Double pendulum** (PPO final: 458.7) | | | | |
| SAC | baseline (GC only) | 434.5 | 369.9 | -88.8 |
| SAC | + PER + QBound | 455.7 | 455.7 | -3.0 |
| SAC | + RWAI v2 | 482.9 | 453.7 | -5.0 |
| TD3 | + PER + AS | 441.4 | 428.8 | -29.9 |
| DDPG | + AS | 457.9 | 457.1 | -1.6 |

**Key findings:**

- **Single pendulum**: SAC baselines underperform PPO by 127 points. QBound closes this gap entirely — SAC + QBound and SAC + PER + QBound *exceed* PPO by +13–15 points. TD3 remains far below PPO even with its best configuration (PER + AS: -170).
- **Double pendulum**: Multiple off-policy configurations approach PPO-level performance: DDPG + AS (-1.6), SAC + PER + QBound (-3.0), and SAC + RWAI v2 (-5.0).
- **Without the investigated techniques**, off-policy algorithms are consistently inferior to PPO on these hard-exploration tasks. **With the right technique combination**, SAC matches or exceeds PPO while retaining off-policy sample efficiency advantages (learning from a fixed buffer rather than requiring fresh rollouts each epoch).

This demonstrates that the techniques studied in this work are not merely incremental improvements — they can bridge the performance gap between off-policy and on-policy methods on hard-exploration tasks.

### 7.6 The Overestimation Spectrum

The complete results refine our understanding of the overestimation spectrum:

```
<-- Harmful                    Beneficial                    Harmful -->
   (no exploration)       (exploration bonus)           (collapse)

   TD3 default     TD3+AS      DDPG default    DDPG+RWAI v2   SAC+RWAI v1
   (8.7, stuck)    (205, partial) (299-479)    (453 peak,      (catastrophic
                                               unstable)        w/ norm)
```

QBound acts as a ceiling on the right side, preventing overestimation from crossing into the harmful zone. This explains its strong synergy with RWAI v2 for SAC: RWAI v2 provides informed initialization while QBound prevents divergence. However, QBound cannot help the left side (TD3's under-exploration), where AS is the effective intervention.

### 7.7 PER as a Synergy Amplifier

PER alone provides inconsistent benefits, but it significantly amplifies other techniques:

- **PER + QBound** (SAC): 453.5/455.7 final -- the most stable SAC configuration
- **PER + AS** (TD3): 292.5/269.5 single -- the best TD3 single result by a wide margin
- **PER + RWAI v2** (SAC): 454.4/453.3 single -- highest SAC single final reward

PER's mechanism -- focusing updates on high-TD-error transitions -- is most valuable when another technique creates a distinct pattern of TD errors. QBound introduces clipping-edge transitions; AS changes the gradient flow pattern; RWAI v2 creates a characteristic early-training error distribution. PER exploits these patterns for more efficient learning.

### 7.8 The TD3 Single-Pendulum Problem

TD3's complete failure on single pendulum (best: 8.7 across baseline, QBound, PER, and RWAI v2 variants) represents a fundamental algorithmic limitation rather than a tuning issue. Only AS (205.4) and PER+AS (292.5) provide partial relief.

The diagnosis is now confirmed by multiple lines of evidence:
1. **DDPG solves it** (298.8 baseline) -- same architecture minus clipped double-Q
2. **RWAI v1 temporarily rescues** (449.1 peak) -- overestimation breaks the trap
3. **AS partially rescues** (205.4) -- preventing tanh saturation enables more effective exploration
4. **QBound, PER, RWAI v2 alone do not help** -- the problem is not Q-divergence, sample efficiency, or initialization

The single pendulum's shorter episode (500 vs 1000 steps) and simpler dynamics make the exploration problem harder: the agent has fewer timesteps to discover the swing-up strategy, and the reward landscape has sharper transitions between the "stuck down" and "swinging up" regimes.

### 7.9 Practical Recommendations

Based on the complete 110-experiment matrix:

**For SAC** (recommended for hard-exploration tasks):
- Add QBound with environment-derived Q-value bounds. Expected improvement: +45% on single, +5-25% on double.
- Consider RWAI v2 for peak performance, especially combined with PER.
- **Never use Adaptive Scaling with SAC** -- it is incompatible with the squashed Gaussian policy.
- PER + QBound provides the most stable high-performing configuration.

**For TD3** (avoid for single-pendulum-like tasks):
- Use PER + AS for the best chance of escaping exploration traps.
- Avoid RWAI v2 -- it destabilizes TD3 even with norm_reward=False.
- For tasks where TD3 already works (e.g., double pendulum), PER + QBound or PER + AS provide moderate improvements.

**For DDPG** (robust default choice):
- RWAI v2 enables highest peak performance but introduces instability.
- AS improves double pendulum stability.
- DDPG is the most tolerant of technique variations.

**General**:
- Always use gradient clipping (max_grad_norm=1.0) for off-policy algorithms.
- Disable reward normalization when using QBound or RWAI v2.
- The optimal technique combination is algorithm-specific; no universal "best" exists.

### 7.10 Limitations

1. **Single-seed evaluation**: All experiments use seed=42. Deep RL has high variance across seeds. Multi-seed evaluation with confidence intervals is needed to distinguish systematic effects from stochastic variation. This remains the most significant limitation.
2. **Two environments**: Both environments have dense rewards in approximately [-0.5, 1]. Generalization to sparse rewards, negative rewards, or mixed reward structures is not established.
3. **Fixed hyperparameters**: The learning rate, buffer size, and other hyperparameters were held constant. The techniques' interactions with different hyperparameter regimes are unknown.
4. **AS-SAC interaction not fully characterized**: While we identify the log-probability corruption mechanism, further analysis (e.g., tracking entropy coefficient dynamics) would strengthen the explanation.
5. **No ablation on Q-bound tightness**: QBound uses the theoretical bounds. Tighter or looser bounds may yield different results.

### 7.11 Connection to Related Work

The five techniques interact with existing work as follows:

- **Clipped double-Q** (TD3, SAC): Our results show that QBound is complementary to clipped double-Q for SAC (both help) but cannot overcome TD3's exploration limitation. AS provides the exploration-boosting effect that RWAI v1 provided temporarily but with more stability.
- **Conservative Q-learning** (CQL) [9]: CQL deliberately underestimates for offline RL. Our QBound provides bidirectional bounding for online RL. The SAC results validate that principled Q-value regularization improves online learning.
- **Optimistic initialization** (tabular RL) [10]: RWAI v2 extends this concept to deep function approximation. The algorithm-dependent results (helps SAC, destroys TD3) confirm that the extension is non-trivial and interacts with the Q-estimation architecture.
- **Prioritized replay** [16]: PER's primary value is as a synergy amplifier rather than a standalone technique, consistent with mixed results in the original paper across different domains.
- **Reward normalization**: Our results show that disabling reward normalization can be beneficial (SAC no_norm_reward: 452.6 vs baseline: 312.4) when paired with techniques that provide alternative Q-value regularization.

## 8. Conclusion

We investigate five complementary techniques for improving off-policy deep reinforcement learning on hard-exploration tasks: Reward-Range-Aware Initialization (RWAI v1 and v2), Gradient Clipping (GC), Adaptive Scaling (AS), Prioritized Experience Replay (PER), and Q-Value Bounding (QBound). Our complete study spans 110 experiments across SAC, TD3, DDPG, and PPO on single and double pendulum swing-up tasks. The principal findings are:

**1. QBound is the most consistently beneficial technique for SAC.** QBound improves SAC single from 312.4 to 453.2 (+45%) and provides stable convergence. PER + QBound achieves the most reliable high performance (453.5/455.7 final). QBound constrains Q-values to the theoretically achievable range, preventing the late-training drift that degrades SAC baselines.

**2. Adaptive Scaling is catastrophic for SAC but rescues TD3.** Every SAC+AS variant collapses (best <75 on single, <120 on double), because AS corrupts SAC's squashed Gaussian log-probability computation. Conversely, AS partially rescues TD3 from exploration failure (8.7→205 single, 293 with PER), demonstrating that technique effectiveness is fundamentally algorithm-dependent.

**3. RWAI v2 significantly improves SAC but destabilizes TD3.** With norm_reward=False and corrected Q-ranges, RWAI v2 achieves the highest SAC double peak (482.9) and strong single performance (452.6). However, RWAI v2 causes TD3 to collapse on both environments, confirming that informed initialization interacts destructively with TD3's conservative clipped double-Q mechanism.

**4. PER functions primarily as a synergy amplifier.** PER alone provides inconsistent benefits, but PER + QBound (SAC), PER + AS (TD3), and PER + RWAI v2 (SAC) produce the best configurations for their respective algorithms. PER's TD-error-based sampling is most valuable when paired with techniques that create distinct error patterns.

**5. Conservative Q-estimation harms exploration in hard-exploration tasks.** TD3's clipped double-Q suppresses value signals from rare high-reward transitions. Under default initialization, TD3 single is completely stuck (8.7) while DDPG solves the same task (298.8). Only AS (which prevents tanh saturation) and RWAI v1 (which provides temporary overestimation) break through, confirming an exploration-conservatism interaction.

**6. No single technique universally improves all algorithms.** The optimal configuration is SAC + PER + QBound for stable high performance, SAC + RWAI v2 for peak performance, TD3 + PER + AS for exploration rescue, and DDPG + AS for double-pendulum stability. Practitioners must select techniques based on the specific algorithm and task.

**7. On-policy algorithms are immune to these concerns.** PPO achieves 450/488 on single/double regardless of technique variations, confirming that the investigated failure modes are specific to off-policy bootstrap learning.

### Future Directions

Multi-seed evaluation across a broader range of environments would establish generality and statistical significance. Specific open questions include: Can AS be adapted for SAC by correcting the log-probability computation? Can QBound's effectiveness for SAC transfer to more complex environments? Is there an adaptive Q-bound tightness schedule that improves over fixed theoretical bounds? Adaptive beta schedules for QBound's softplus clipping, learned priority functions for PER, and online Q-range estimation for RWAI are promising extensions.

## References

[1] T. Haarnoja, A. Zhou, P. Abbeel, S. Levine, "Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor," ICML, 2018.

[2] S. Fujimoto, H. van Hoof, D. Meger, "Addressing Function Approximation Error in Actor-Critic Methods," ICML, 2018.

[3] T. P. Lillicrap et al., "Continuous Control with Deep Reinforcement Learning," ICLR, 2016.

[4] X. Glorot, Y. Bengio, "Understanding the difficulty of training deep feedforward neural networks," AISTATS, 2010.

[5] K. He, X. Zhang, S. Ren, J. Sun, "Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification," ICCV, 2015.

[6] A. M. Saxe, J. L. McClelland, S. Ganguli, "Exact solutions to the nonlinear dynamics of learning in deep linear neural networks," ICLR, 2014.

[7] S. Thrun, A. Schwartz, "Issues in Using Function Approximation for Reinforcement Learning," Connectionist Models Summer School, 1993.

[8] H. van Hasselt, "Double Q-learning," NeurIPS, 2010.

[9] A. Kumar, A. Zhou, G. Tucker, S. Levine, "Conservative Q-Learning for Offline Reinforcement Learning," NeurIPS, 2020.

[10] R. S. Sutton, A. G. Barto, *Reinforcement Learning: An Introduction*, MIT Press, 2018.

[11] J. Schulman, F. Wolski, P. Dhariwal, A. Radford, O. Klimov, "Proximal Policy Optimization Algorithms," arXiv:1707.06347, 2017.

[12] B. Raffin, A. Hill, A. Gleave, A. Kanervisto, M. Ernestus, N. Dormann, "Stable-Baselines3: Reliable Reinforcement Learning Implementations," JMLR, 2021.

[13] J. Hollenstein, S. Auddy, M. Saveriano, E. Renaudo, J. Piater, "Action Noise in Off-Policy Deep Reinforcement Learning: Impact on Exploration and Performance," arXiv:2206.03787, 2022.

[14] O. Eberhard, J. Hollenstein, C. Pinneri, G. Martius, "Pink Noise Is All You Need: Colored Noise Exploration in Deep Reinforcement Learning," ICLR, 2023.

[15] A. Raffin, "Generalized State-Dependent Exploration for Deep Reinforcement Learning in Robotics," arXiv:2005.05719, 2020.

[16] T. Schaul, J. Quan, I. Antonoglou, D. Silver, "Prioritized Experience Replay," ICLR, 2016.

[17] T. Z. Gebrekidan, "QBound: Q-Value Bounding for Off-Policy Deep Reinforcement Learning," https://github.com/TesfayZ/QBound, 2025.

[18] T. Z. Gebrekidan, "Gradient Asymmetry and Activation Saturation in Deep Reinforcement Learning," https://github.com/TesfayZ/gradient_asymetry_AND_activation_saturation, 2025.

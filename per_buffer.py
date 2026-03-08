"""
Prioritized Experience Replay (PER) buffer for Stable Baselines3.

Implements the proportional prioritization variant from:
  Schaul et al., "Prioritized Experience Replay", ICLR 2016.

Key components:
  - SumTree: O(log n) data structure for sampling proportional to priorities
  - PrioritizedReplayBuffer: Drop-in replacement for SB3's ReplayBuffer
    with priority-based sampling and importance-sampling weight correction

PER hyperparameters:
  - alpha (0.6): Priority exponent — 0 = uniform, 1 = full prioritization
  - beta (0.4 → 1.0): IS weight exponent — annealed linearly to 1.0
"""

import numpy as np
from typing import Optional

from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.type_aliases import ReplayBufferSamples
from stable_baselines3.common.vec_env import VecNormalize


class SumTree:
    """Binary tree where each parent is the sum of its children.

    Supports O(log n) operations for:
      - update(idx, priority): Set priority for a leaf node
      - sample(): Sample a leaf index proportional to its priority
      - total(): Return the sum of all priorities

    Storage layout: tree[0] is root, tree[capacity-1..2*capacity-2] are leaves.
    Leaf i corresponds to tree[capacity - 1 + i].
    """

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float64)
        self._min_tree = np.full(2 * capacity - 1, float("inf"), dtype=np.float64)

    def update(self, leaf_idx: int, priority: float) -> None:
        """Set the priority of leaf_idx and propagate up."""
        tree_idx = leaf_idx + self.capacity - 1
        self.tree[tree_idx] = priority
        self._min_tree[tree_idx] = priority

        while tree_idx > 0:
            tree_idx = (tree_idx - 1) // 2
            left = 2 * tree_idx + 1
            right = 2 * tree_idx + 2
            self.tree[tree_idx] = self.tree[left] + self.tree[right]
            self._min_tree[tree_idx] = min(self._min_tree[left], self._min_tree[right])

    def sample(self, value: float) -> int:
        """Find the leaf index for a cumulative sum value."""
        tree_idx = 0
        while True:
            left = 2 * tree_idx + 1
            right = 2 * tree_idx + 2
            if left >= len(self.tree):
                # At a leaf
                break
            if value <= self.tree[left]:
                tree_idx = left
            else:
                value -= self.tree[left]
                tree_idx = right
        return tree_idx - (self.capacity - 1)

    @property
    def total(self) -> float:
        return self.tree[0]

    @property
    def min(self) -> float:
        return self._min_tree[0]


class PrioritizedReplayBuffer(ReplayBuffer):
    """Replay buffer with proportional prioritization.

    New transitions are added with max priority to ensure they are sampled
    at least once. Priorities are updated externally after computing TD errors.

    Args:
        buffer_size: Max number of transitions.
        observation_space: Observation space.
        action_space: Action space.
        alpha: Priority exponent (0 = uniform, 1 = full prioritization).
        beta_init: Initial IS weight exponent.
        beta_final: Final IS weight exponent (reached at end of training).
        total_timesteps: Total training steps (for beta annealing schedule).
        **kwargs: Passed to ReplayBuffer.
    """

    def __init__(
        self,
        buffer_size: int,
        observation_space,
        action_space,
        alpha: float = 0.6,
        beta_init: float = 0.4,
        beta_final: float = 1.0,
        total_timesteps: int = 500_000,
        **kwargs,
    ):
        super().__init__(buffer_size, observation_space, action_space, **kwargs)
        self.alpha = alpha
        self.beta_init = beta_init
        self.beta_final = beta_final
        self.total_timesteps = total_timesteps

        self._sum_tree = SumTree(self.buffer_size)
        self._max_priority = 1.0
        self._env_timestep = 0

        # Seeded RNG for reproducible sampling (seed set by SB3 via global np.random)
        self._rng = np.random.default_rng(np.random.randint(0, 2**31))

        # Store last sampled indices for priority updates
        self._last_batch_inds: Optional[np.ndarray] = None

    def set_env_timestep(self, timestep: int) -> None:
        """Update the current environment timestep for beta annealing."""
        self._env_timestep = timestep

    @property
    def beta(self) -> float:
        """Current beta value (linearly annealed from beta_init to beta_final)."""
        fraction = min(self._env_timestep / max(self.total_timesteps, 1), 1.0)
        return self.beta_init + fraction * (self.beta_final - self.beta_init)

    def add(self, obs, next_obs, action, reward, done, infos) -> None:
        """Add a transition with max priority."""
        idx = self.pos
        super().add(obs, next_obs, action, reward, done, infos)
        # Assign max priority so new transitions are sampled at least once
        self._sum_tree.update(idx, self._max_priority ** self.alpha)

    def sample(
        self, batch_size: int, env: Optional[VecNormalize] = None
    ) -> ReplayBufferSamples:
        """Sample transitions proportional to their priorities.

        Returns standard ReplayBufferSamples. IS weights are stored
        in self._last_is_weights for use in the training loop.
        """
        current_size = self.buffer_size if self.full else self.pos
        if current_size == 0:
            raise ValueError("Cannot sample from empty buffer")

        total = self._sum_tree.total
        if total <= 0:
            # Fallback to uniform if tree is empty
            batch_inds = self._rng.integers(0, current_size, size=batch_size)
            self._last_batch_inds = batch_inds
            self._last_is_weights = np.ones(batch_size, dtype=np.float32)
            return self._get_samples(batch_inds, env=env)

        # Stratified sampling: divide [0, total) into batch_size segments
        segment = total / batch_size
        batch_inds = np.zeros(batch_size, dtype=np.int64)
        priorities = np.zeros(batch_size, dtype=np.float64)

        for i in range(batch_size):
            low = segment * i
            high = segment * (i + 1)
            value = self._rng.uniform(low, high)
            idx = self._sum_tree.sample(value)
            # Clamp to valid range
            idx = np.clip(idx, 0, current_size - 1)
            batch_inds[i] = idx
            priorities[i] = max(self._sum_tree.tree[idx + self._sum_tree.capacity - 1], 1e-8)

        # Compute importance-sampling weights
        probs = priorities / total
        beta = self.beta

        # w_i = (N * P(i))^(-beta)
        weights = (current_size * probs) ** (-beta)
        # Normalize by max weight for stability
        weights /= weights.max()

        self._last_batch_inds = batch_inds
        self._last_is_weights = weights.astype(np.float32)

        return self._get_samples(batch_inds, env=env)

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray) -> None:
        """Update priorities based on TD errors.

        Args:
            indices: Buffer indices (from self._last_batch_inds).
            td_errors: Absolute TD errors for each transition.
        """
        # Small constant to prevent zero priorities
        priorities = np.abs(td_errors) + 1e-6

        for idx, priority in zip(indices, priorities):
            self._sum_tree.update(int(idx), priority ** self.alpha)
            self._max_priority = max(self._max_priority, priority)

        # Decay max_priority towards the current batch max to prevent
        # a single outlier from inflating all future initial priorities
        batch_max = priorities.max()
        self._max_priority = max(0.95 * self._max_priority, batch_max)

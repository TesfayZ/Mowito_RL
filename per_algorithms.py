"""
Algorithm subclasses with Prioritized Experience Replay (PER) support.

Overrides train() for SAC, TD3, and DDPG to:
  1. Extract importance-sampling (IS) weights from the PER buffer
  2. Weight the critic loss by IS weights (corrects sampling bias)
  3. Compute TD errors and update buffer priorities

DDPG inherits from TD3 in SB3, so DDPGWithPER inherits from TD3WithPER.
"""

import numpy as np
import torch as th
from torch.nn import functional as F

from stable_baselines3 import SAC, TD3, DDPG
from stable_baselines3.common.utils import polyak_update

from per_buffer import PrioritizedReplayBuffer
from utils import sync_scaler_buffers


class SACWithPER(SAC):
    """SAC with Prioritized Experience Replay.

    Modifies the critic loss to be weighted by importance-sampling weights
    and updates buffer priorities based on TD errors after each gradient step.
    """

    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        self.policy.set_training_mode(True)
        optimizers = [self.actor.optimizer, self.critic.optimizer]
        if self.ent_coef_optimizer is not None:
            optimizers += [self.ent_coef_optimizer]
        self._update_learning_rate(optimizers)

        ent_coef_losses, ent_coefs = [], []
        actor_losses, critic_losses = [], []

        for gradient_step in range(gradient_steps):
            # Sample replay buffer (PER: returns prioritized samples)
            is_per_buffer = isinstance(self.replay_buffer, PrioritizedReplayBuffer)
            if is_per_buffer:
                self.replay_buffer.set_env_timestep(self.num_timesteps)
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)
            discounts = replay_data.discounts if replay_data.discounts is not None else self.gamma

            # Get IS weights from PER buffer
            if is_per_buffer:
                is_weights = th.as_tensor(
                    self.replay_buffer._last_is_weights, device=self.device
                ).reshape(-1, 1)
            else:
                is_weights = th.ones((batch_size, 1), device=self.device)

            if self.use_sde:
                self.actor.reset_noise()

            # Action by current actor for sampled state
            actions_pi, log_prob = self.actor.action_log_prob(replay_data.observations)
            log_prob = log_prob.reshape(-1, 1)

            # Entropy coefficient
            ent_coef_loss = None
            if self.ent_coef_optimizer is not None and self.log_ent_coef is not None:
                ent_coef = th.exp(self.log_ent_coef.detach())
                assert isinstance(self.target_entropy, float)
                ent_coef_loss = -(self.log_ent_coef * (log_prob + self.target_entropy).detach()).mean()
                ent_coef_losses.append(ent_coef_loss.item())
            else:
                ent_coef = self.ent_coef_tensor
            ent_coefs.append(ent_coef.item())

            if ent_coef_loss is not None and self.ent_coef_optimizer is not None:
                self.ent_coef_optimizer.zero_grad()
                ent_coef_loss.backward()
                self.ent_coef_optimizer.step()

            # Compute target Q-values
            with th.no_grad():
                next_actions, next_log_prob = self.actor.action_log_prob(replay_data.next_observations)
                next_q_values = th.cat(self.critic_target(replay_data.next_observations, next_actions), dim=1)
                next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)
                next_q_values = next_q_values - ent_coef * next_log_prob.reshape(-1, 1)
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * discounts * next_q_values

            # Get current Q-values
            current_q_values = self.critic(replay_data.observations, replay_data.actions)

            # PER-weighted critic loss: weight each sample's squared error by IS weight
            critic_loss = 0.5 * sum(
                (is_weights * (current_q - target_q_values) ** 2).mean()
                for current_q in current_q_values
            )
            assert isinstance(critic_loss, th.Tensor)
            critic_losses.append(critic_loss.item())

            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()

            # Update PER priorities using TD error from first Q-network
            if is_per_buffer:
                with th.no_grad():
                    td_errors = (current_q_values[0] - target_q_values).abs().cpu().numpy().flatten()
                self.replay_buffer.update_priorities(
                    self.replay_buffer._last_batch_inds, td_errors
                )

            # Actor loss (unchanged from standard SAC)
            q_values_pi = th.cat(self.critic(replay_data.observations, actions_pi), dim=1)
            min_qf_pi, _ = th.min(q_values_pi, dim=1, keepdim=True)
            actor_loss = (ent_coef * log_prob - min_qf_pi).mean()
            actor_losses.append(actor_loss.item())

            self.actor.optimizer.zero_grad()
            actor_loss.backward()
            self.actor.optimizer.step()

            # Update target networks
            if gradient_step % self.target_update_interval == 0:
                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
                polyak_update(self.batch_norm_stats, self.batch_norm_stats_target, 1.0)

        self._n_updates += gradient_steps
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/ent_coef", np.mean(ent_coefs))
        self.logger.record("train/actor_loss", np.mean(actor_losses))
        self.logger.record("train/critic_loss", np.mean(critic_losses))
        if len(ent_coef_losses) > 0:
            self.logger.record("train/ent_coef_loss", np.mean(ent_coef_losses))


class TD3WithPER(TD3):
    """TD3 with Prioritized Experience Replay.

    Modifies the critic loss to be weighted by importance-sampling weights
    and updates buffer priorities based on TD errors after each gradient step.
    """

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        self.policy.set_training_mode(True)
        self._update_learning_rate([self.actor.optimizer, self.critic.optimizer])

        actor_losses, critic_losses = [], []
        for _ in range(gradient_steps):
            self._n_updates += 1

            is_per_buffer = isinstance(self.replay_buffer, PrioritizedReplayBuffer)
            if is_per_buffer:
                self.replay_buffer.set_env_timestep(self.num_timesteps)
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)
            discounts = replay_data.discounts if replay_data.discounts is not None else self.gamma

            # Get IS weights from PER buffer
            if is_per_buffer:
                is_weights = th.as_tensor(
                    self.replay_buffer._last_is_weights, device=self.device
                ).reshape(-1, 1)
            else:
                is_weights = th.ones((batch_size, 1), device=self.device)

            # Compute target Q-values
            with th.no_grad():
                noise = replay_data.actions.clone().data.normal_(0, self.target_policy_noise)
                noise = noise.clamp(-self.target_noise_clip, self.target_noise_clip)
                next_actions = (self.actor_target(replay_data.next_observations) + noise).clamp(-1, 1)

                next_q_values = th.cat(self.critic_target(replay_data.next_observations, next_actions), dim=1)
                next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * discounts * next_q_values

            # Get current Q-values
            current_q_values = self.critic(replay_data.observations, replay_data.actions)

            # PER-weighted critic loss
            critic_loss = sum(
                (is_weights * (current_q - target_q_values) ** 2).mean()
                for current_q in current_q_values
            )
            assert isinstance(critic_loss, th.Tensor)
            critic_losses.append(critic_loss.item())

            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()

            # Update PER priorities
            if is_per_buffer:
                with th.no_grad():
                    td_errors = (current_q_values[0] - target_q_values).abs().cpu().numpy().flatten()
                self.replay_buffer.update_priorities(
                    self.replay_buffer._last_batch_inds, td_errors
                )

            # Delayed policy updates
            if self._n_updates % self.policy_delay == 0:
                actor_loss = -self.critic.q1_forward(
                    replay_data.observations, self.actor(replay_data.observations)
                ).mean()
                actor_losses.append(actor_loss.item())

                self.actor.optimizer.zero_grad()
                actor_loss.backward()
                self.actor.optimizer.step()

                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
                polyak_update(self.actor.parameters(), self.actor_target.parameters(), self.tau)
                polyak_update(self.critic_batch_norm_stats, self.critic_batch_norm_stats_target, 1.0)
                polyak_update(self.actor_batch_norm_stats, self.actor_batch_norm_stats_target, 1.0)

                # Sync adaptive scaler buffers if present
                sync_scaler_buffers(self.actor, self.actor_target)

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        if len(actor_losses) > 0:
            self.logger.record("train/actor_loss", np.mean(actor_losses))
        self.logger.record("train/critic_loss", np.mean(critic_losses))


class DDPGWithPER(TD3WithPER):
    """DDPG with Prioritized Experience Replay.

    DDPG is TD3 with policy_delay=1, no target noise clipping, and a single
    Q-network. Inheriting from TD3WithPER gives us PER support automatically.
    Uses the same constructor defaults as SB3's DDPG.
    """
    pass

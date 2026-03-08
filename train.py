"""
Training script for Cart-Pole Swing-Up RL agents.

Usage:
    python train.py --experiment ppo_single
    python train.py --experiment sac_single
    python train.py --experiment ppo_double
    python train.py --experiment sac_double
    python train.py --all   # Train all experiments sequentially
"""

import argparse
import copy
import json
import os
import sys
import time
import numpy as np
import torch
import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import (
    EvalCallback,
    CheckpointCallback,
    BaseCallback,
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure

# Register custom environments
import envs  # noqa: F401

from configs import EXPERIMENTS
from gc_algorithms import (
    SACWithGC, TD3WithGC, DDPGWithGC,
    SACWithPERAndGC, TD3WithPERAndGC, DDPGWithPERAndGC,
)
from qbound import (
    SACWithGCAndQB, TD3WithGCAndQB, DDPGWithGCAndQB,
    SACWithPERAndGCAndQB, TD3WithPERAndGCAndQB, DDPGWithPERAndGCAndQB,
)
from per_buffer import PrioritizedReplayBuffer


class SyncNormCallback(BaseCallback):
    """Syncs VecNormalize stats from training env to eval env before each eval."""

    def __init__(self, train_env, eval_env, verbose=0):
        super().__init__(verbose)
        self.train_env = train_env
        self.eval_env = eval_env

    def _on_step(self) -> bool:
        # Sync running stats from training to eval env
        self.eval_env.obs_rms = self.train_env.obs_rms
        self.eval_env.ret_rms = self.train_env.ret_rms
        return True


class RewardLoggingCallback(BaseCallback):
    """Logs episode reward statistics during training."""

    def __init__(self, log_freq=10_000, verbose=0):
        super().__init__(verbose)
        self.log_freq = log_freq
        self.episode_rewards = []

    def _on_step(self) -> bool:
        # Collect episode rewards from the monitor wrapper
        infos = self.locals.get("infos", [])
        for info in infos:
            if "episode" in info:
                self.episode_rewards.append(info["episode"]["r"])

        if self.num_timesteps % self.log_freq == 0 and self.episode_rewards:
            recent = self.episode_rewards[-100:]
            mean_r = np.mean(recent)
            std_r = np.std(recent)
            max_r = np.max(recent)
            print(
                f"  Step {self.num_timesteps}: "
                f"mean_reward={mean_r:.2f} +/- {std_r:.2f}, "
                f"max_reward={max_r:.2f}, "
                f"episodes={len(self.episode_rewards)}"
            )
        return True


def make_env(env_id, rank=0, seed=0):
    """Create a monitored environment instance."""
    def _init():
        env = gym.make(env_id)
        env = Monitor(env)
        env.reset(seed=seed + rank)
        return env
    return _init


def train_experiment(name, exp_config, results_dir="results"):
    """Train a single experiment."""
    seed = exp_config.get("seed", 42)

    # Set torch deterministic mode for reproducibility
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    print(f"\n{'='*60}")
    print(f"Training: {name}")
    print(f"  Algorithm: {exp_config['algo']}")
    print(f"  Environment: {exp_config['env_id']}")
    print(f"  Timesteps: {exp_config['timesteps']:,}")
    print(f"  Parallel envs: {exp_config['n_envs']}")
    print(f"  Seed: {seed}")
    print(f"{'='*60}\n")

    # Create output directory
    exp_dir = os.path.join(results_dir, name)
    os.makedirs(exp_dir, exist_ok=True)

    # Create vectorized environment
    n_envs = exp_config["n_envs"]
    env_id = exp_config["env_id"]

    if n_envs > 1:
        env = SubprocVecEnv([make_env(env_id, i, seed=seed) for i in range(n_envs)])
    else:
        env = DummyVecEnv([make_env(env_id, 0, seed=seed)])

    # Wrap with VecNormalize for observation/reward normalization
    norm_reward = exp_config.get("norm_reward", True)
    env = VecNormalize(env, norm_obs=True, norm_reward=norm_reward, clip_obs=10.0)

    # Create evaluation environment (separate, synced normalization stats)
    eval_env = DummyVecEnv([make_env(env_id, seed=seed + 1000)])
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, clip_obs=10.0)
    eval_env.training = False  # don't update stats during eval

    # Select algorithm
    algo_config = copy.deepcopy(exp_config["config"])
    algo_map = {
        "PPO": PPO,
        # All off-policy experiments use GC (gradient clipping).
        # 4 class types: GC, PER+GC, GC+QB, PER+GC+QB.
        "SAC_GC": SACWithGC,
        "TD3_GC": TD3WithGC,
        "DDPG_GC": DDPGWithGC,
        "SAC_PER_GC": SACWithPERAndGC,
        "TD3_PER_GC": TD3WithPERAndGC,
        "DDPG_PER_GC": DDPGWithPERAndGC,
        "SAC_GC_QB": SACWithGCAndQB,
        "TD3_GC_QB": TD3WithGCAndQB,
        "DDPG_GC_QB": DDPGWithGCAndQB,
        "SAC_PER_GC_QB": SACWithPERAndGCAndQB,
        "TD3_PER_GC_QB": TD3WithPERAndGCAndQB,
        "DDPG_PER_GC_QB": DDPGWithPERAndGCAndQB,
    }
    AlgoClass = algo_map[exp_config["algo"]]

    # Pass max_grad_norm to GC algorithm classes
    max_grad_norm = exp_config.get("max_grad_norm")
    if max_grad_norm is not None:
        algo_config["max_grad_norm"] = max_grad_norm

    # Pass QBound parameters to QB algorithm classes
    qbound_config = exp_config.get("qbound")
    if qbound_config is not None:
        algo_config["qbound_min"] = qbound_config["qbound_min"]
        algo_config["qbound_max"] = qbound_config["qbound_max"]

    # Create PER buffer if specified
    per_config = exp_config.get("per")
    if per_config is not None:
        algo_config["replay_buffer_class"] = PrioritizedReplayBuffer
        algo_config["replay_buffer_kwargs"] = {
            "alpha": per_config["alpha"],
            "beta_init": per_config["beta_init"],
            "beta_final": per_config["beta_final"],
            "total_timesteps": exp_config["timesteps"],
        }

    # Configure logger
    logger = configure(exp_dir, ["stdout", "csv"])

    # Create model with seed for reproducibility
    model = AlgoClass(env=env, seed=seed, **algo_config)
    model.set_logger(logger)

    # Callbacks
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(exp_dir, "best_model"),
        log_path=os.path.join(exp_dir, "eval_logs"),
        eval_freq=max(10_000 // n_envs, 1000),
        n_eval_episodes=10,
        deterministic=True,
        verbose=1,
        warn=False,
    )
    checkpoint_callback = CheckpointCallback(
        save_freq=max(50_000 // n_envs, 5000),
        save_path=os.path.join(exp_dir, "checkpoints"),
        name_prefix=name,
    )
    sync_callback = SyncNormCallback(env, eval_env)
    reward_callback = RewardLoggingCallback(log_freq=10_000)

    # Save experiment config for reproducibility
    config_record = {
        "name": name,
        "algo": exp_config["algo"],
        "env_id": exp_config["env_id"],
        "timesteps": exp_config["timesteps"],
        "seed": seed,
        "n_envs": n_envs,
        "norm_reward": norm_reward,
        "max_grad_norm": exp_config.get("max_grad_norm"),
        "per": exp_config.get("per"),
        "qbound": exp_config.get("qbound"),
    }
    with open(os.path.join(exp_dir, "experiment_config.json"), "w") as f:
        json.dump(config_record, f, indent=2)

    # Train
    start = time.time()
    model.save(os.path.join(exp_dir, "initial_model"))

    try:
        model.learn(
            total_timesteps=exp_config["timesteps"],
            callback=[sync_callback, eval_callback, checkpoint_callback, reward_callback],
            progress_bar=True,
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted! Saving current model...")

    elapsed = time.time() - start
    print(f"\nTraining completed in {elapsed:.0f}s ({elapsed/60:.1f}min)")

    # Save final model and normalization stats
    model.save(os.path.join(exp_dir, "final_model"))
    env.save(os.path.join(exp_dir, "vec_normalize.pkl"))

    # Log final stats
    if reward_callback.episode_rewards:
        recent = reward_callback.episode_rewards[-100:]
        print(f"Final mean reward (last 100 episodes): {np.mean(recent):.2f}")
        print(f"Total episodes: {len(reward_callback.episode_rewards)}")

    env.close()
    eval_env.close()

    return exp_dir


def main():
    parser = argparse.ArgumentParser(description="Train RL agents for cart-pole swing-up")
    parser.add_argument(
        "--experiment", "-e",
        choices=list(EXPERIMENTS.keys()),
        help="Experiment to run",
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Run all experiments sequentially",
    )
    parser.add_argument(
        "--results-dir", default="results",
        help="Directory to save results (default: results)",
    )
    args = parser.parse_args()

    if not args.experiment and not args.all:
        parser.print_help()
        sys.exit(1)

    if args.all:
        experiments = list(EXPERIMENTS.keys())
    else:
        experiments = [args.experiment]

    for name in experiments:
        # Skip experiments that already have a final model with matching config
        exp_dir = os.path.join(args.results_dir, name)
        final_path = os.path.join(exp_dir, "final_model.zip")
        config_path = os.path.join(exp_dir, "experiment_config.json")
        if os.path.exists(final_path):
            # Verify the existing model was trained with matching timesteps
            if os.path.exists(config_path):
                try:
                    with open(config_path) as f:
                        saved_config = json.load(f)
                    if saved_config.get("timesteps") == EXPERIMENTS[name]["timesteps"]:
                        print(f"\nSkipping {name} — final_model.zip exists with matching config")
                        continue
                    else:
                        print(f"\nRe-training {name} — config mismatch (timesteps: "
                              f"{saved_config.get('timesteps')} vs {EXPERIMENTS[name]['timesteps']})")
                except (json.JSONDecodeError, KeyError):
                    print(f"\nRe-training {name} — corrupt experiment_config.json")
            else:
                print(f"\nSkipping {name} — final_model.zip exists (no config to verify)")
                continue
        train_experiment(name, EXPERIMENTS[name], args.results_dir)

    print("\nAll training complete!")
    print(f"Results saved to: {os.path.abspath(args.results_dir)}")


if __name__ == "__main__":
    main()

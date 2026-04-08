"""
Evaluation and video recording script for trained RL agents.

Usage:
    python test.py --experiment ppo_single --episodes 10
    python test.py --experiment sac_single --record
    python test.py --experiment ppo_double --record --random-init
"""

import argparse
import os
import sys
import numpy as np
import gymnasium as gym

from stable_baselines3 import PPO, SAC, TD3, DDPG
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# Register custom environments
import envs  # noqa: F401

from configs import EXPERIMENTS
from per_algorithms import SACWithPER, TD3WithPER, DDPGWithPER
from gc_algorithms import (
    SACWithGC, TD3WithGC, DDPGWithGC,
    SACWithPERAndGC, TD3WithPERAndGC, DDPGWithPERAndGC,
)
from qbound import (
    SACWithGCAndQB, TD3WithGCAndQB, DDPGWithGCAndQB,
    SACWithPERAndGCAndQB, TD3WithPERAndGCAndQB, DDPGWithPERAndGCAndQB,
)

_ALGO_MAP = {
    "PPO": PPO,
    "SAC": SAC,
    "TD3": TD3,
    "DDPG": DDPG,
    "SAC_PER": SACWithPER,
    "TD3_PER": TD3WithPER,
    "DDPG_PER": DDPGWithPER,
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


def load_model(exp_name, results_dir="results", use_best=True):
    """Load a trained model and its normalization stats."""
    exp_dir = os.path.join(results_dir, exp_name)
    exp_config = EXPERIMENTS[exp_name]

    # Load model
    AlgoClass = _ALGO_MAP[exp_config["algo"]]
    if use_best:
        model_path = os.path.join(exp_dir, "best_model", "best_model.zip")
        if not os.path.exists(model_path):
            model_path = os.path.join(exp_dir, "final_model.zip")
    else:
        model_path = os.path.join(exp_dir, "final_model.zip")

    print(f"Loading model from: {model_path}")
    model = AlgoClass.load(model_path)

    # Load VecNormalize stats
    norm_path = os.path.join(exp_dir, "vec_normalize.pkl")
    return model, norm_path, exp_config


def evaluate(exp_name, n_episodes=10, results_dir="results", record=False,
             random_init=False, render=False):
    """Evaluate a trained agent and optionally record videos."""
    model, norm_path, exp_config = load_model(exp_name, results_dir)
    env_id = exp_config["env_id"]

    # Create environment
    if record:
        video_dir = os.path.join(results_dir, exp_name, "videos")
        os.makedirs(video_dir, exist_ok=True)
        render_mode = "rgb_array"
    elif render:
        render_mode = "human"
    else:
        render_mode = None

    eval_seed = 12345  # Fixed seed for reproducible evaluation

    def make_eval_env():
        env = gym.make(env_id, render_mode=render_mode)
        if record:
            env = gym.wrappers.RecordVideo(
                env,
                video_folder=video_dir,
                episode_trigger=lambda ep: True,  # record every episode
                name_prefix=f"{exp_name}_eval",
            )
        return env

    env = DummyVecEnv([make_eval_env])
    env.seed(eval_seed)

    # Load normalization stats
    if os.path.exists(norm_path):
        env = VecNormalize.load(norm_path, env)
        env.training = False  # don't update stats during evaluation
        env.norm_reward = False
    else:
        print("Warning: No VecNormalize stats found, using raw observations")

    # Run evaluation episodes
    episode_rewards = []
    episode_lengths = []
    balance_times = []
    successes = []

    for ep in range(n_episodes):
        reset_options = {"random_init": True} if random_init else {}
        obs = env.reset(options=reset_options)
        done = False
        total_reward = 0.0
        steps = 0
        balanced_steps = 0
        first_balanced = None

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, dones, infos = env.step(action)
            total_reward += reward[0]
            steps += 1
            done = dones[0]

            info = infos[0]
            if info.get("is_balanced", False):
                balanced_steps += 1
                if first_balanced is None:
                    first_balanced = steps

        episode_rewards.append(total_reward)
        episode_lengths.append(steps)

        # Consider success if balanced for 50+ consecutive-ish steps
        success = balanced_steps > 50
        successes.append(success)

        if first_balanced is not None:
            balance_times.append(first_balanced * 0.02)  # convert to seconds

        status = "BALANCED" if success else "failed"
        print(
            f"  Episode {ep+1}/{n_episodes}: "
            f"reward={total_reward:.2f}, "
            f"steps={steps}, "
            f"balanced_steps={balanced_steps}, "
            f"[{status}]"
        )

    env.close()

    # Print summary
    print(f"\n{'='*50}")
    print(f"Evaluation Summary: {exp_name}")
    print(f"{'='*50}")
    print(f"  Episodes: {n_episodes}")
    print(f"  Mean reward: {np.mean(episode_rewards):.2f} +/- {np.std(episode_rewards):.2f}")
    print(f"  Mean length: {np.mean(episode_lengths):.0f} steps")
    print(f"  Success rate: {np.mean(successes)*100:.1f}%")
    if balance_times:
        print(f"  Mean time to balance: {np.mean(balance_times):.2f}s")
    if record:
        print(f"  Videos saved to: {video_dir}")
    print()

    return {
        "mean_reward": np.mean(episode_rewards),
        "std_reward": np.std(episode_rewards),
        "success_rate": np.mean(successes),
        "mean_length": np.mean(episode_lengths),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained RL agents")
    parser.add_argument(
        "--experiment", "-e",
        choices=list(EXPERIMENTS.keys()),
        required=True,
        help="Experiment to evaluate",
    )
    parser.add_argument(
        "--episodes", "-n", type=int, default=10,
        help="Number of evaluation episodes (default: 10)",
    )
    parser.add_argument(
        "--results-dir", default="results",
        help="Directory containing results (default: results)",
    )
    parser.add_argument(
        "--record", action="store_true",
        help="Record videos of evaluation episodes",
    )
    parser.add_argument(
        "--random-init", action="store_true",
        help="Use random initial angles (not just hanging down)",
    )
    parser.add_argument(
        "--render", action="store_true",
        help="Render environment in a window (human mode)",
    )
    args = parser.parse_args()

    evaluate(
        args.experiment,
        n_episodes=args.episodes,
        results_dir=args.results_dir,
        record=args.record,
        random_init=args.random_init,
        render=args.render,
    )


if __name__ == "__main__":
    main()

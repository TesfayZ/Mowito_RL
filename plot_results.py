"""
Generate training curves and comparison plots.

Usage:
    python plot_results.py                    # Plot all available experiments
    python plot_results.py --experiments ppo_single sac_single
"""

import argparse
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from stable_baselines3.common.results_plotter import load_results, ts2xy

from configs import EXPERIMENTS


def smooth(values, weight=0.9):
    """Exponential moving average for smoothing curves."""
    smoothed = []
    last = values[0] if len(values) > 0 else 0
    for v in values:
        last = weight * last + (1 - weight) * v
        smoothed.append(last)
    return smoothed


def plot_training_curve(exp_name, results_dir="results", ax=None, color=None,
                        show_raw=False):
    """Plot training reward curve for a single experiment.

    Args:
        show_raw: If True, plot raw (unsmoothed) data as transparent background.
                  If False (default), only plot the smoothed curve for cleaner legends.
    """
    exp_dir = os.path.join(results_dir, exp_name)
    monitor_dir = exp_dir

    # Try to load from monitor logs
    csv_path = os.path.join(exp_dir, "progress.csv")
    if os.path.exists(csv_path):
        data = np.genfromtxt(csv_path, delimiter=",", names=True)
        timesteps_col = None
        reward_col = None
        for name in data.dtype.names:
            if "timestep" in name.lower() or name == "time/total_timesteps":
                timesteps_col = name
            if "reward" in name.lower() and "mean" in name.lower():
                reward_col = name
            if name == "rollout/ep_rew_mean":
                reward_col = name
                timesteps_col = "time/total_timesteps"

        if timesteps_col and reward_col:
            timesteps = data[timesteps_col]
            rewards = data[reward_col]
            # Filter out NaN
            mask = ~np.isnan(rewards)
            timesteps = timesteps[mask]
            rewards = rewards[mask]

            if ax is None:
                fig, ax = plt.subplots(1, 1, figsize=(10, 6))

            if show_raw:
                ax.plot(timesteps, rewards, alpha=0.3, color=color)
            ax.plot(timesteps, smooth(rewards, 0.9), linewidth=2,
                    label=exp_name, color=color)
            return True

    # Try monitor files
    try:
        timesteps, rewards = ts2xy(load_results(exp_dir), "timesteps")
        if len(timesteps) > 0:
            if ax is None:
                fig, ax = plt.subplots(1, 1, figsize=(10, 6))
            if show_raw:
                ax.plot(timesteps, rewards, alpha=0.2, color=color)
            ax.plot(timesteps, smooth(rewards, 0.9), linewidth=2,
                    label=exp_name, color=color)
            return True
    except Exception:
        pass

    # Try eval logs
    eval_path = os.path.join(exp_dir, "eval_logs", "evaluations.npz")
    if os.path.exists(eval_path):
        data = np.load(eval_path)
        timesteps = data["timesteps"]
        results = data["results"]
        mean_rewards = np.mean(results, axis=1)

        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(10, 6))

        std_rewards = np.std(results, axis=1)
        ax.plot(timesteps, mean_rewards, linewidth=2, label=exp_name, color=color)
        ax.fill_between(timesteps,
                        mean_rewards - std_rewards,
                        mean_rewards + std_rewards,
                        alpha=0.2, color=color)
        return True

    print(f"Warning: No training data found for {exp_name}")
    return False


def plot_all(experiments=None, results_dir="results"):
    """Generate all comparison plots."""
    if experiments is None:
        experiments = [name for name in EXPERIMENTS
                       if os.path.exists(os.path.join(results_dir, name))]

    if not experiments:
        print("No experiment results found. Run train.py first.")
        return

    colors = plt.cm.tab10(np.linspace(0, 1, len(experiments)))

    # ── Single Pendulum Comparison ────────────────────────────────────────
    single_exps = [e for e in experiments if "single" in e]
    if single_exps:
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        for i, exp in enumerate(single_exps):
            plot_training_curve(exp, results_dir, ax, colors[experiments.index(exp)])
        ax.set_xlabel("Timesteps", fontsize=12)
        ax.set_ylabel("Episode Reward", fontsize=12)
        ax.set_title("Single Pendulum Swing-Up: Training Curves", fontsize=14)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        path = os.path.join(results_dir, "training_curves_single.png")
        fig.savefig(path, dpi=150)
        print(f"Saved: {path}")
        plt.close(fig)

    # ── Double Pendulum Comparison ────────────────────────────────────────
    double_exps = [e for e in experiments if "double" in e]
    if double_exps:
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        for i, exp in enumerate(double_exps):
            plot_training_curve(exp, results_dir, ax, colors[experiments.index(exp)])
        ax.set_xlabel("Timesteps", fontsize=12)
        ax.set_ylabel("Episode Reward", fontsize=12)
        ax.set_title("Double Pendulum Swing-Up: Training Curves", fontsize=14)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        path = os.path.join(results_dir, "training_curves_double.png")
        fig.savefig(path, dpi=150)
        print(f"Saved: {path}")
        plt.close(fig)

    # ── Combined Plot ─────────────────────────────────────────────────────
    fig, ax = plt.subplots(1, 1, figsize=(12, 7))
    for i, exp in enumerate(experiments):
        plot_training_curve(exp, results_dir, ax, colors[i])
    ax.set_xlabel("Timesteps", fontsize=12)
    ax.set_ylabel("Episode Reward", fontsize=12)
    ax.set_title("Cart-Pole Swing-Up: All Training Curves", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(results_dir, "training_curves_all.png")
    fig.savefig(path, dpi=150)
    print(f"Saved: {path}")
    plt.close(fig)


def plot_v1_vs_v2_comparison(results_dir="results"):
    """Generate side-by-side v1 vs v2 RWAI comparison plots.

    Produces a 4x2 grid (SAC/TD3/DDPG/PPO × single/double) with baseline,
    RWAI v1, and RWAI v2 eval curves overlaid per subplot.

    Output: results/rwai_v1_vs_v2_comparison.png
    """
    algos = ["sac", "td3", "ddpg"]
    envs = ["single", "double"]
    algo_labels = {"sac": "SAC", "td3": "TD3", "ddpg": "DDPG"}
    env_labels = {"single": "Single Pendulum", "double": "Double Pendulum"}

    # Colors for each variant
    variant_colors = {
        "baseline": "#1f77b4",  # blue
        "v1": "#ff7f0e",       # orange
        "v2": "#2ca02c",       # green
    }

    fig, axes = plt.subplots(3, 2, figsize=(16, 16))

    for row, algo in enumerate(algos):
        for col, env in enumerate(envs):
            ax = axes[row, col]

            # Experiment names for this cell
            baseline_name = f"{algo}_{env}"
            v1_name = f"{algo}_{env}_rwinit"
            v2_name = f"{algo}_{env}_rwinit_v2"

            variants = {
                "baseline": baseline_name,
                "v1": v1_name,
                "v2": v2_name,
            }

            has_data = False
            for label, exp_name in variants.items():
                exp_dir = os.path.join(results_dir, exp_name)
                if os.path.exists(exp_dir):
                    if plot_training_curve(exp_name, results_dir, ax, variant_colors[label]):
                        has_data = True

            ax.set_title(f"{algo_labels[algo]} — {env_labels[env]}", fontsize=13)
            ax.set_xlabel("Timesteps", fontsize=10)
            ax.set_ylabel("Episode Reward", fontsize=10)
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)

            if not has_data:
                ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                        ha="center", va="center", fontsize=14, color="gray")

    fig.suptitle("RWAI v1 (bias-only) vs v2 (scaled weights) Comparison",
                 fontsize=16, fontweight="bold", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    path = os.path.join(results_dir, "rwai_v1_vs_v2_comparison.png")
    fig.savefig(path, dpi=150)
    print(f"Saved: {path}")
    plt.close(fig)

    # Print summary table
    _print_summary_table(results_dir, algos, envs, algo_labels,
                         ["baseline", "v1", "v2"],
                         {"baseline": "", "v1": "_rwinit", "v2": "_rwinit_v2"})


def plot_full_comparison(results_dir="results"):
    """Generate full comparison: baseline vs RWAI v1 vs v2 vs PER variants.

    Produces a 3x2 grid (SAC/TD3/DDPG × single/double) with all available
    variants overlaid. PPO is excluded since PER is off-policy only.

    Output: results/full_comparison.png
    """
    algos = ["sac", "td3", "ddpg"]
    envs = ["single", "double"]
    algo_labels = {"sac": "SAC", "td3": "TD3", "ddpg": "DDPG"}
    env_labels = {"single": "Single Pendulum", "double": "Double Pendulum"}

    variant_colors = {
        "baseline": "#1f77b4",      # blue
        "v1": "#ff7f0e",            # orange
        "v2": "#2ca02c",            # green
        "baseline+PER": "#d62728",  # red
        "v2+PER": "#9467bd",        # purple
    }
    variant_suffixes = {
        "baseline": "",
        "v1": "_rwinit",
        "v2": "_rwinit_v2",
        "baseline+PER": "_per",
        "v2+PER": "_rwinit_v2_per",
    }

    fig, axes = plt.subplots(3, 2, figsize=(16, 16))

    for row, algo in enumerate(algos):
        for col, env in enumerate(envs):
            ax = axes[row, col]

            has_data = False
            for label, suffix in variant_suffixes.items():
                exp_name = f"{algo}_{env}{suffix}"
                exp_dir = os.path.join(results_dir, exp_name)
                if os.path.exists(exp_dir):
                    if plot_training_curve(exp_name, results_dir, ax, variant_colors[label]):
                        has_data = True

            ax.set_title(f"{algo_labels[algo]} — {env_labels[env]}", fontsize=13)
            ax.set_xlabel("Timesteps", fontsize=10)
            ax.set_ylabel("Episode Reward", fontsize=10)
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)

            if not has_data:
                ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                        ha="center", va="center", fontsize=14, color="gray")

    fig.suptitle("Full Comparison: Baseline vs RWAI v1 vs v2 vs PER",
                 fontsize=16, fontweight="bold", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    path = os.path.join(results_dir, "full_comparison.png")
    fig.savefig(path, dpi=150)
    print(f"Saved: {path}")
    plt.close(fig)

    # Print summary table
    _print_summary_table(results_dir, algos, envs, algo_labels,
                         list(variant_suffixes.keys()), variant_suffixes)


def plot_gcas_comparison(results_dir="results"):
    """Generate GCAS comparison: baseline vs GCAS vs RWAI+GCAS vs PER+GCAS.

    Produces a 3x2 grid (SAC/TD3/DDPG x single/double) with all GCAS
    variants overlaid against baselines.

    Output: results/gcas_comparison.png
    """
    algos = ["sac", "td3", "ddpg"]
    envs = ["single", "double"]
    algo_labels = {"sac": "SAC", "td3": "TD3", "ddpg": "DDPG"}
    env_labels = {"single": "Single Pendulum", "double": "Double Pendulum"}

    variant_colors = {
        "baseline": "#1f77b4",      # blue
        "AS": "#2ca02c",            # green
        "RWAI+AS": "#ff7f0e",       # orange
        "PER+AS": "#d62728",        # red
    }
    variant_suffixes = {
        "baseline": "",
        "AS": "_as",
        "RWAI+AS": "_rwinit_v2_as",
        "PER+AS": "_per_as",
    }

    fig, axes = plt.subplots(3, 2, figsize=(16, 16))

    for row, algo in enumerate(algos):
        for col, env in enumerate(envs):
            ax = axes[row, col]

            has_data = False
            for label, suffix in variant_suffixes.items():
                exp_name = f"{algo}_{env}{suffix}"
                exp_dir = os.path.join(results_dir, exp_name)
                if os.path.exists(exp_dir):
                    if plot_training_curve(exp_name, results_dir, ax, variant_colors[label]):
                        has_data = True

            ax.set_title(f"{algo_labels[algo]} — {env_labels[env]}", fontsize=13)
            ax.set_xlabel("Timesteps", fontsize=10)
            ax.set_ylabel("Episode Reward", fontsize=10)
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)

            if not has_data:
                ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                        ha="center", va="center", fontsize=14, color="gray")

    fig.suptitle("GCAS Comparison: Baseline vs Grad Clip + Adaptive Scaling",
                 fontsize=16, fontweight="bold", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    path = os.path.join(results_dir, "gcas_comparison.png")
    fig.savefig(path, dpi=150)
    print(f"Saved: {path}")
    plt.close(fig)

    # Print summary table
    _print_summary_table(results_dir, algos, envs, algo_labels,
                         list(variant_suffixes.keys()), variant_suffixes)


def plot_qbound_comparison(results_dir="results"):
    """Generate QBound comparison: baseline vs QBound vs PER+QBound.

    Produces a 3x2 grid (SAC/TD3/DDPG x single/double).

    Output: results/qbound_comparison.png
    """
    algos = ["sac", "td3", "ddpg"]
    envs = ["single", "double"]
    algo_labels = {"sac": "SAC", "td3": "TD3", "ddpg": "DDPG"}
    env_labels = {"single": "Single Pendulum", "double": "Double Pendulum"}

    variant_colors = {
        "baseline": "#1f77b4",      # blue
        "QBound": "#2ca02c",        # green
        "PER+QBound": "#d62728",    # red
    }
    variant_suffixes = {
        "baseline": "",
        "QBound": "_qbound",
        "PER+QBound": "_per_qbound",
    }

    fig, axes = plt.subplots(3, 2, figsize=(16, 16))

    for row, algo in enumerate(algos):
        for col, env in enumerate(envs):
            ax = axes[row, col]

            has_data = False
            for label, suffix in variant_suffixes.items():
                exp_name = f"{algo}_{env}{suffix}"
                exp_dir = os.path.join(results_dir, exp_name)
                if os.path.exists(exp_dir):
                    if plot_training_curve(exp_name, results_dir, ax, variant_colors[label]):
                        has_data = True

            ax.set_title(f"{algo_labels[algo]} — {env_labels[env]}", fontsize=13)
            ax.set_xlabel("Timesteps", fontsize=10)
            ax.set_ylabel("Episode Reward", fontsize=10)
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)

            if not has_data:
                ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                        ha="center", va="center", fontsize=14, color="gray")

    fig.suptitle("QBound Comparison: Baseline vs QBound vs PER+QBound",
                 fontsize=16, fontweight="bold", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    path = os.path.join(results_dir, "qbound_comparison.png")
    fig.savefig(path, dpi=150)
    print(f"Saved: {path}")
    plt.close(fig)

    _print_summary_table(results_dir, algos, envs, algo_labels,
                         list(variant_suffixes.keys()), variant_suffixes)


def _print_summary_table(results_dir, algos, envs, algo_labels,
                         variant_names, variant_suffixes):
    """Print a summary table with best/final rewards for all variants."""
    print(f"\n{'='*90}")
    print("Summary Table: Best and Final Mean Eval Rewards")
    print(f"{'='*90}")

    header = f"{'Algo':<6} {'Env':<8}"
    for v in variant_names:
        header += f" | {v:>16s}"
    print(header)
    print("-" * len(header))

    for algo in algos:
        for env in envs:
            row = f"{algo_labels.get(algo, algo):<6} {env:<8}"
            for v in variant_names:
                suffix = variant_suffixes[v]
                exp_name = f"{algo}_{env}{suffix}"
                eval_path = os.path.join(results_dir, exp_name, "eval_logs", "evaluations.npz")
                if os.path.exists(eval_path):
                    data = np.load(eval_path)
                    mean_rewards = np.mean(data["results"], axis=1)
                    best = np.max(mean_rewards)
                    final = mean_rewards[-1] if len(mean_rewards) > 0 else float("nan")
                    row += f" | {best:7.1f}/{final:7.1f}"
                else:
                    row += f" | {'N/A':>16s}"
            print(row)

    print(f"{'='*90}")
    print("Format: best_reward / final_reward\n")


def main():
    parser = argparse.ArgumentParser(description="Plot training results")
    parser.add_argument(
        "--experiments", nargs="+",
        help="Specific experiments to plot (default: all available)",
    )
    parser.add_argument(
        "--results-dir", default="results",
        help="Directory containing results",
    )
    parser.add_argument(
        "--v1-vs-v2", action="store_true",
        help="Generate RWAI v1 vs v2 comparison plots",
    )
    parser.add_argument(
        "--full-comparison", action="store_true",
        help="Generate full comparison plots (baseline/v1/v2/PER)",
    )
    parser.add_argument(
        "--gcas-comparison", action="store_true",
        help="Generate GCAS comparison plots (baseline/GCAS/RWAI+GCAS/PER+GCAS)",
    )
    parser.add_argument(
        "--qbound-comparison", action="store_true",
        help="Generate QBound comparison plots (baseline/QBound/PER+QBound)",
    )
    args = parser.parse_args()

    if args.v1_vs_v2:
        plot_v1_vs_v2_comparison(args.results_dir)
    elif args.full_comparison:
        plot_full_comparison(args.results_dir)
    elif args.gcas_comparison:
        plot_gcas_comparison(args.results_dir)
    elif args.qbound_comparison:
        plot_qbound_comparison(args.results_dir)
    else:
        plot_all(args.experiments, args.results_dir)


if __name__ == "__main__":
    main()

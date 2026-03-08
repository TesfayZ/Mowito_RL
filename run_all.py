"""
Comprehensive experiment runner for RL stabilization ablation study.

Organizes and runs all experiments in logical groups:
  - PPO baselines (on-policy, no contributions needed)
  - Off-policy baselines (SAC, TD3, DDPG with GC only, no contributions)
  - Single contributions (one of RWAI v2, PER, AS, QBound)
  - Two-way combinations
  - Three-way combinations
  - Full combination (all four contributions)
  - Legacy RWAI v1 (retained for reference)

All off-policy experiments include gradient clipping (GC) as standard.
The four contributions ablated combinatorially are:
  RWAI v2, PER, AS (Adaptive Scaling), QBound.

Usage:
    python run_all.py                        # Run all experiments
    python run_all.py --group ppo            # Run only PPO baselines
    python run_all.py --group baselines      # Run only off-policy baselines
    python run_all.py --group single         # Single contributions only
    python run_all.py --group two-way        # Two-way combinations
    python run_all.py --group three-way      # Three-way combinations
    python run_all.py --group full           # Full combination (all 4)
    python run_all.py --group contributions  # All non-baseline experiments
    python run_all.py --group legacy         # Legacy RWAI v1
    python run_all.py --algo sac             # Only SAC experiments
    python run_all.py --env single           # Only single pendulum
    python run_all.py --dry-run              # Show plan without executing
    python run_all.py --skip-existing        # Skip experiments with results
"""

import argparse
import itertools
import os
import sys
import time

from configs import EXPERIMENTS
from train import train_experiment


# ── Experiment Groups ────────────────────────────────────────────────────────
# Generated programmatically to match configs.py naming.

_ALGOS = ["sac", "td3", "ddpg"]
_ENVS = ["single", "double"]
_CONTRIBUTIONS = ["rwinit_v2", "per", "as", "qbound"]

PPO = ["ppo_single", "ppo_double"]

# Off-policy baselines — GC only, no contributions
BASELINES = [f"{a}_{e}" for a in _ALGOS for e in _ENVS]

# Legacy RWAI v1 (bias-only, retained for reference)
LEGACY_RWINIT_V1 = [f"{a}_{e}_rwinit" for a in _ALGOS for e in _ENVS]


def _make_group(n_contributions):
    """Generate experiment names with exactly n contributions enabled."""
    names = []
    for combo in itertools.combinations(_CONTRIBUTIONS, n_contributions):
        for a in _ALGOS:
            for e in _ENVS:
                name = f"{a}_{e}_{'_'.join(combo)}"
                names.append(name)
    return names


SINGLE_CONTRIBUTIONS = _make_group(1)
TWO_WAY_COMBINATIONS = _make_group(2)
THREE_WAY_COMBINATIONS = _make_group(3)
FULL_COMBINATION = _make_group(4)

# Group mapping
GROUPS = {
    "ppo": PPO,
    "baselines": BASELINES,
    "legacy": LEGACY_RWINIT_V1,
    "single": SINGLE_CONTRIBUTIONS,
    "two-way": TWO_WAY_COMBINATIONS,
    "three-way": THREE_WAY_COMBINATIONS,
    "full": FULL_COMBINATION,
    "contributions": (SINGLE_CONTRIBUTIONS + TWO_WAY_COMBINATIONS +
                      THREE_WAY_COMBINATIONS + FULL_COMBINATION),
    "all": (PPO + BASELINES + SINGLE_CONTRIBUTIONS + TWO_WAY_COMBINATIONS +
            THREE_WAY_COMBINATIONS + FULL_COMBINATION + LEGACY_RWINIT_V1),
}


def filter_experiments(experiments, algo=None, env=None):
    """Filter experiment list by algorithm and/or environment."""
    filtered = experiments
    if algo:
        algo_lower = algo.lower()
        filtered = [e for e in filtered if e.startswith(algo_lower + "_")]
    if env:
        env_lower = env.lower()
        filtered = [e for e in filtered if f"_{env_lower}" in e or e.endswith(f"_{env_lower}")]
    return filtered


def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive experiment runner for RL stabilization ablation study",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Experiment Matrix:
  PPO: 2 experiments (on-policy baseline, no contributions)
  Off-policy baselines: 6 experiments (SAC/TD3/DDPG x single/double, GC only)
  Single contributions: 24 experiments (4 contributions x 3 algos x 2 envs)
  Two-way combinations: 36 experiments (C(4,2)=6 pairs x 3 algos x 2 envs)
  Three-way combinations: 24 experiments (C(4,3)=4 triplets x 3 algos x 2 envs)
  Full combination: 6 experiments (all 4 contributions x 3 algos x 2 envs)
  Legacy RWAI v1: 6 experiments (for reference)
  Total: 104 experiments

All off-policy experiments include gradient clipping (max_grad_norm=1.0).
Contributions ablated combinatorially:
  RWAI v2    — Reward-range-aware critic initialization (scaled weights)
  PER        — Prioritized Experience Replay
  AS         — Adaptive Scaling (prevents tanh saturation)
  QBound     — Q-value bounding to theoretical range
        """,
    )
    parser.add_argument(
        "--group", choices=list(GROUPS.keys()), default="all",
        help="Experiment group to run (default: all)",
    )
    parser.add_argument(
        "--algo", choices=["ppo", "sac", "td3", "ddpg"],
        help="Filter by algorithm",
    )
    parser.add_argument(
        "--env", choices=["single", "double"],
        help="Filter by environment",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show experiment plan without executing",
    )
    parser.add_argument(
        "--skip-existing", action="store_true",
        help="Skip experiments that already have final_model.zip",
    )
    parser.add_argument(
        "--results-dir", default="results",
        help="Directory to save results (default: results)",
    )
    args = parser.parse_args()

    # Build experiment list
    experiments = GROUPS[args.group]
    experiments = filter_experiments(experiments, args.algo, args.env)

    if args.skip_existing:
        experiments = [
            e for e in experiments
            if not os.path.exists(os.path.join(args.results_dir, e, "final_model.zip"))
        ]

    # Validate all experiments exist in config
    missing = [e for e in experiments if e not in EXPERIMENTS]
    if missing:
        print(f"ERROR: Unknown experiments: {missing}")
        sys.exit(1)

    if not experiments:
        print("No experiments to run (all filtered out or already complete).")
        sys.exit(0)

    # Display plan
    print(f"\n{'='*70}")
    print(f"RL Stabilization Experiment Runner")
    print(f"{'='*70}")
    print(f"  Group: {args.group}")
    if args.algo:
        print(f"  Algorithm filter: {args.algo}")
    if args.env:
        print(f"  Environment filter: {args.env}")
    print(f"  Experiments to run: {len(experiments)}")
    print(f"  Results directory: {os.path.abspath(args.results_dir)}")
    print(f"{'='*70}\n")

    # Group experiments by category for display
    categories = [
        ("PPO Baselines", PPO),
        ("Off-Policy Baselines (GC only)", BASELINES),
        ("Single Contributions", SINGLE_CONTRIBUTIONS),
        ("Two-Way Combinations", TWO_WAY_COMBINATIONS),
        ("Three-Way Combinations", THREE_WAY_COMBINATIONS),
        ("Full Combination (all 4)", FULL_COMBINATION),
        ("Legacy RWAI v1", LEGACY_RWINIT_V1),
    ]

    for cat_name, cat_list in categories:
        active = [e for e in experiments if e in cat_list]
        if active:
            print(f"  {cat_name} ({len(active)}):")
            for e in active:
                print(f"    - {e}")
            print()

    if args.dry_run:
        print("DRY RUN — no experiments will be executed.")
        return

    # Run experiments
    total = len(experiments)
    completed = 0
    failed = []
    start_time = time.time()

    for i, name in enumerate(experiments, 1):
        print(f"\n{'#'*70}")
        print(f"# [{i}/{total}] {name}")
        print(f"# Completed: {completed}, Failed: {len(failed)}")
        print(f"{'#'*70}")

        try:
            train_experiment(name, EXPERIMENTS[name], args.results_dir)
            completed += 1
        except KeyboardInterrupt:
            print(f"\nInterrupted at experiment {name}.")
            print(f"Completed {completed}/{total} experiments.")
            if failed:
                print(f"Failed: {failed}")
            sys.exit(1)
        except Exception as e:
            print(f"\nERROR in {name}: {e}")
            failed.append(name)

    elapsed = time.time() - start_time
    hours = elapsed / 3600

    print(f"\n{'='*70}")
    print(f"All experiments finished!")
    print(f"  Total time: {hours:.1f} hours ({elapsed:.0f}s)")
    print(f"  Completed: {completed}/{total}")
    if failed:
        print(f"  Failed ({len(failed)}):")
        for f in failed:
            print(f"    - {f}")
    print(f"  Results: {os.path.abspath(args.results_dir)}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()

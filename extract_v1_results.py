"""Extract RWAI v1 and vanilla baseline results for paper table update."""

import os
import numpy as np

ALGOS = ["sac", "td3", "ddpg"]
ENVS = ["single", "double"]
RESULTS_DIR = "results"


def get_eval_results(exp_name):
    """Extract best and final mean eval rewards from evaluations.npz."""
    eval_path = os.path.join(RESULTS_DIR, exp_name, "eval_logs", "evaluations.npz")
    if not os.path.exists(eval_path):
        return None, None

    data = np.load(eval_path)
    results = data["results"]  # shape: (n_evals, n_episodes)
    mean_rewards = results.mean(axis=1)

    best = float(mean_rewards.max())
    final = float(mean_rewards[-1])
    return round(best, 1), round(final, 1)


print("\n=== RWAI v1 Legacy Table (for paper.tex Table 2) ===\n")
print(f"{'Algorithm':<10} {'Env':<8} {'Default Best':>13} {'Default Final':>14} {'RWAI v1 Best':>13} {'RWAI v1 Final':>14}")
print("-" * 75)

latex_rows = []
for algo in ALGOS:
    for env in ENVS:
        vanilla_name = f"{algo}_{env}_vanilla"
        v1_name = f"{algo}_{env}_rwinit"

        d_best, d_final = get_eval_results(vanilla_name)
        v1_best, v1_final = get_eval_results(v1_name)

        algo_upper = algo.upper()
        env_cap = env.capitalize()

        if d_best is not None and v1_best is not None:
            print(f"{algo_upper:<10} {env_cap:<8} {d_best:>13.1f} {d_final:>14.1f} {v1_best:>13.1f} {v1_final:>14.1f}")
            # Generate LaTeX row
            v1_final_str = f"${v1_final}$" if v1_final < 0 else str(v1_final)
            d_final_str = f"${d_final}$" if d_final < 0 else str(d_final)
            latex_rows.append(
                f"{algo_upper} & {env_cap} & {d_best} & {d_final_str} & {v1_best} & {v1_final_str} \\\\"
            )
        else:
            status = []
            if d_best is None:
                status.append(f"vanilla missing")
            if v1_best is None:
                status.append(f"v1 missing")
            print(f"{algo_upper:<10} {env_cap:<8} {'N/A':>13} {'N/A':>14} {'N/A':>13} {'N/A':>14}  ({', '.join(status)})")

print("\n=== LaTeX table rows (copy to paper.tex) ===\n")
for row in latex_rows:
    print(row)

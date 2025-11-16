import numpy as np
import scipy.linalg
import navlie as nav
import timeit
import matplotlib.pyplot as plt
import cProfile

from navlie import GaussianResultList, State, Measurement
from navlie import monte_carlo, MonteCarloResult
from typing import List, Tuple
import pickle
import os

SHOW_FIGS = True
SAVE_FIGS = True
MC_TRIALS = 50
TRIAL_TIME = 4
MEAS_NOISE = "nlos"

if MEAS_NOISE == "student_t":
    results_path = f"./data/results/monte_carlo/mc_{MC_TRIALS}_trials_{TRIAL_TIME}s_{MEAS_NOISE}.pkl"
elif MEAS_NOISE == "nlos":
    results_path = f"./data/results/monte_carlo/mc_{MC_TRIALS}_trials_{TRIAL_TIME}s_{MEAS_NOISE}_1.pkl"
else:
    results_path = f"./data/results/monte_carlo/mc_{MC_TRIALS}_trials_{TRIAL_TIME}s.pkl"

# Load data
with open(
    results_path,
    "rb",
) as f:
    mc_data = pickle.load(f)

results_map: MonteCarloResult = mc_data["results_map"]
results_gmm: MonteCarloResult = mc_data["results_gmm"]
results_gvi: MonteCarloResult = mc_data["results_gvi"]

# Plotting parameters
plt.rc("text", usetex=True)
plt.rc("font", family="serif", size=14)
plt.rc("lines", linewidth=2)
plt.rc("axes", grid=True)
plt.rc("grid", linestyle="--")

fig, ax = nav.plot_nees(
    results=results_map,
    confidence_interval=None,
    label="MAP (Cauchy)",
    color="tab:blue",
    normalize=True,
)
fig, ax = nav.plot_nees(
    results=results_gmm,
    ax=ax,
    confidence_interval=None,
    label="MAP (GMM)",
    color="tab:purple",
    normalize=True,
)
fig, ax = nav.plot_nees(
    results=results_gvi,
    ax=ax,
    confidence_interval=0.997,
    label="ESGVI",
    color="tab:orange",
    normalize=True,
)
ax.set_ylabel(r"Normalized Squared Mahalanobis Distance")
ax.set_xlabel("Time (s)")
# ax.set_title(f"aNEES {MC_TRIALS} trials.")
if SAVE_FIGS:
    if MEAS_NOISE == "student_t":
        plt.savefig(
            f"/home/astirl/Documents/courses/assignments/mech_642/gvi_ws/figs/monte_carlo/se2_aNEES_{MC_TRIALS}_{int(TRIAL_TIME)}s_st.pdf"
        )
    elif MEAS_NOISE == "nlos":
        plt.savefig(
            f"/home/astirl/Documents/courses/assignments/mech_642/gvi_ws/figs/monte_carlo/se2_aNEES_{MC_TRIALS}_{int(TRIAL_TIME)}s_nlos.pdf"
        )
    else:
        plt.savefig(
            f"/home/astirl/Documents/courses/assignments/mech_642/gvi_ws/figs/monte_carlo/se2_aNEES_{MC_TRIALS}_{int(TRIAL_TIME)}s.pdf"
        )

plt.show()

print("Estimation Performance")
print("   Method    | RMSE (rad) |  RMSE (m)  |  aNEES")
print("------------------------------------------------")
print(
    f" MAP (Cauchy)|  {np.mean(results_map.rmse[:,0]):.4f}   |  {np.mean(results_map.rmse[:,1:]):.4f}   |  {(np.mean(results_map.average_nees/results_map.dof)):.3f}"
)
print(
    f" MAP (GMM)   |  {np.mean(results_gmm.rmse[:,0]):.4f}   |  {np.mean(results_gmm.rmse[:,1:]):.4f}   |  {(np.mean(results_gmm.average_nees/results_gmm.dof)):.3f}"
)

print(
    f" ESGVI       |  {np.mean(results_gvi.rmse[:,0]):.4f}   |  {np.mean(results_gvi.rmse[:,1:]):.4f}   |  {(np.mean(results_gvi.average_nees/results_gvi.dof)):.3f}"
)
print(" ----------------------------------------------")

if results_gvi.num_trials < 15:
    fig, axs = plt.subplots(3, 2)
    axs: List[plt.Axes] = axs
    for result in results_gvi.trial_results:
        nav.plot_error(result, axs=axs)

    fig.suptitle("Estimation error")
    plt.tight_layout()
    plt.show()

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# -------------------------------------------------------------
# Collect RMSE and aNEES data into a DataFrame
# -------------------------------------------------------------
data = []

# Assuming rmse has shape (N, 3) → [orientation, x, y]
# and average_nees is a 1D array (per trial or per time step)

methods = [
    ("MAP (Cauchy)", results_map),
    ("MAP (GMM)", results_gmm),
    ("ESGVI", results_gvi),
]

for name, res in methods:
    # Orientation and position RMSE per sample
    rmse_rad = res.rmse[:, 0]
    rmse_pos = np.mean(res.rmse[:, 1:], axis=1)  # average x/y per sample
    anees = res.average_nees / res.dof

    data.extend(
        [{"Method": name, "Metric": "RMSE (rad)", "Values": v} for v in rmse_rad]
    )
    data.extend([{"Method": name, "Metric": "RMSE (m)", "Values": v} for v in rmse_pos])
    data.extend([{"Method": name, "Metric": "aNEES", "Values": v} for v in anees])

df = pd.DataFrame(data)

# -------------------------------------------------------------
# Define consistent colors per method
# -------------------------------------------------------------
method_colors = {
    "MAP (Cauchy)": "tab:blue",
    "MAP (GMM)": "tab:purple",
    "ESGVI": "tab:orange",
}

metrics = ["RMSE (rad)", "RMSE (m)", "aNEES"]

# -------------------------------------------------------------
# Create boxplots for each metric
# -------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=False)

for ax, metric in zip(axes, metrics):
    subset = df[df["Metric"] == metric]
    methods = list(method_colors.keys())
    grouped = [subset[subset["Method"] == m]["Values"].values for m in methods]

    bp = ax.boxplot(
        grouped,
        patch_artist=True,
        tick_labels=methods,
        showmeans=False,
        showfliers=False,
        medianprops={"color": "black", "linewidth": 1.2},
        boxprops={"linewidth": 1.2},
        whiskerprops={"linewidth": 1.2},
        capprops={"linewidth": 1.2},
    )

    # Apply consistent colors
    for patch, m in zip(bp["boxes"], methods):
        patch.set_facecolor(method_colors[m])
        patch.set_alpha(0.5)

    # ax.set_title(metric, fontsize=11)
    plt.setp(ax.get_xticklabels(), rotation=15, ha="center")
    ax.set_xlabel("")
    ax.grid(True, linestyle="--", alpha=0.4)
    if metric == "RMSE (rad)":
        ax.set_ylabel(f"Orientation {metric}", fontsize=10)

    elif metric == "RMSE (m)":
        ax.set_ylabel(f"Translation {metric}", fontsize=10)

    else:
        ax.set_ylabel(f"{metric}", fontsize=10)
        # Add horizontal dashed line at y=1 with label
        ax.axhline(
            y=1, color="red", linestyle="--", linewidth=1, label="Expected aNEES"
        )
        confidence_interval = 0.997
        # Add legend only once per subplot
        ci_label = f"${(confidence_interval * 100):.1f}\%$ conf. bounds"

        # Compute NEES confidence bounds (normalized by s if needed)
        upper_bound = (
            results_map.nees_upper_bound(confidence_interval) / results_map.dof
        )
        lower_bound = (
            results_map.nees_lower_bound(confidence_interval) / results_map.dof
        )
        # Plot bounds as dashed lines
        ax.axhline(
            upper_bound[0],
            color="k",
            linestyle="--",
            linewidth=1,
            label=ci_label,
        )
        ax.axhline(
            lower_bound[0],
            color="k",
            linestyle="--",
            linewidth=1,
        )
        ax.legend(loc="upper left", fontsize=8)

plt.tight_layout()
plt.savefig(f"./figs/monte_carlo/boxplot_sim.pdf")
plt.show()

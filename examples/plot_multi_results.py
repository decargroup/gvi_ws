import numpy as np
import navlie as nav
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from navlie import (
    GaussianResultList,
    State,
    Measurement,
    MonteCarloResult,
    associate_stamps,
)
from typing import List, Tuple
import pickle
import os

DATASET = "multi"  # "se2_sim"
MEAS_NOISE = "skew_laplace"  # "skew_laplace"
fname = "results.pkl"
trials = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

if DATASET == "se2_sim":
    raise NotImplementedError("There aren't multiple trials for this dataset")

elif DATASET == "cluttered":
    results_path = DATASET
    raise NotImplementedError("There aren't multiple trials for this dataset")

elif DATASET == "multi":
    results_path = DATASET

else:
    raise NotImplementedError(f"The {DATASET} dataset doesn't exist.")


SHOW_FIGS = True
SAVE_FIGS = True


# Load data
results_map_list = []
results_gmm_list = []
results_gvi_list = []

for trial in trials:
    fname = f"results_{trial}.pkl"
    with open(f"./data/results/{results_path}/map_{fname}", "rb") as f:
        map_data = pickle.load(f)

    with open(f"./data/results/{results_path}/gmm_{fname}", "rb") as f:
        gmm_data = pickle.load(f)

    with open(f"./data/results/{results_path}/gvi_{fname}", "rb") as f:
        gvi_data = pickle.load(f)

    ground_truth: List[State] = map_data["ground_truth"]
    results_map: GaussianResultList = map_data["results_map"]
    sim_time: float = map_data["sim_time"]
    landmarks = map_data["landmarks"]
    results_gmm: GaussianResultList = gmm_data["results_gmm"]
    results_gvi: GaussianResultList = gvi_data["results_gvi"]
    skew_lambda: float = gvi_data["skew_lambda"]

    results_gvi_list.append(results_gvi)
    results_map_list.append(results_map)
    results_gmm_list.append(results_gmm)


#####################
##### PLOT GVI ######
#####################
# Plotting parameters
plt.rc("text", usetex=True)
plt.rc("font", family="serif", size=14)
plt.rc("lines", linewidth=2)
plt.rc("axes", grid=True)
plt.rc("grid", linestyle="--")


rmse_list_map = []
rmse_rad_list_map = []
anees_list_map = []

rmse_list_gmm = []
rmse_rad_list_gmm = []
anees_list_gmm = []

rmse_list_gvi = []
rmse_rad_list_gvi = []
anees_list_gvi = []
print("Estimation Performance")
# Comparison table
for i in range(len(results_map_list)):
    results_map = results_map_list[i]
    results_gmm = results_gmm_list[i]
    results_gvi = results_gvi_list[i]

    rmse_rad_map = np.sqrt(np.mean(np.square(results_map.error[:, 0])))
    rmse_rad_gmm = np.sqrt(np.mean(np.square(results_gmm.error[:, 0])))
    rmse_rad_gvi = np.sqrt(np.mean(np.square(results_gvi.error[:, 0])))
    rmse_map = np.sqrt(np.mean(np.square(results_map.error[:, 1:])))
    rmse_gmm = np.sqrt(np.mean(np.square(results_gmm.error[:, 1:])))
    rmse_gvi = np.sqrt(np.mean(np.square(results_gvi.error[:, 1:])))
    anees_map = np.mean(results_map.nees / results_map.dof)
    anees_gmm = np.mean(results_gmm.nees / results_gmm.dof)
    anees_gvi = np.mean(results_gvi.nees / results_gvi.dof)

    rmse_list_map.append(rmse_map)
    rmse_rad_list_map.append(rmse_rad_map)
    anees_list_map.append(anees_map)

    rmse_list_gmm.append(rmse_gmm)
    rmse_rad_list_gmm.append(rmse_rad_gmm)
    anees_list_gmm.append(anees_gmm)

    rmse_list_gvi.append(rmse_gvi)
    rmse_rad_list_gvi.append(rmse_rad_gvi)
    anees_list_gvi.append(anees_gvi)

print(f"Estimation Performance for {len(trials)} Trials")
print(" Method | RMSE (rad) |  RMSE (m)  |  aNEES")
print("----------------------------------------")
print(
    f" MAP    |  {np.mean(rmse_rad_list_map):.5f}   |  {np.mean(rmse_list_map):.5f}   |  {np.mean(anees_list_map):.5f}"
)

print(
    f" GMM    |  {np.mean(rmse_rad_list_gmm):.5f}   |  {np.mean(rmse_list_gmm):.5f}   |  {np.mean(anees_list_gmm):.5f}"
)

print(
    f" GVI    |  {np.mean(rmse_rad_list_gvi):.5f}   |  {np.mean(rmse_list_gvi):.5f}   |  {np.mean(anees_list_gvi):.5f}"
)


# Concatenate all results across trajectories
all_errors_map = np.vstack([r.error for r in results_map_list])
all_errors_gmm = np.vstack([r.error for r in results_gmm_list])
all_errors_gvi = np.vstack([r.error for r in results_gvi_list])

# Concatenate raw NEES values (no normalization by DOF)
all_nees_map = np.hstack([r.nees for r in results_map_list])
all_nees_gmm = np.hstack([r.nees for r in results_gmm_list])
all_nees_gvi = np.hstack([r.nees for r in results_gvi_list])

# Compute RMSE (orientation and position)
rmse_rad_map = np.sqrt(np.mean(np.square(all_errors_map[:, 0])))
rmse_rad_gmm = np.sqrt(np.mean(np.square(all_errors_gmm[:, 0])))
rmse_rad_gvi = np.sqrt(np.mean(np.square(all_errors_gvi[:, 0])))

rmse_map = np.sqrt(np.mean(np.square(all_errors_map[:, 1:])))
rmse_gmm = np.sqrt(np.mean(np.square(all_errors_gmm[:, 1:])))
rmse_gvi = np.sqrt(np.mean(np.square(all_errors_gvi[:, 1:])))

# Compute average (non-normalized) NEES
anees_map = np.mean(all_nees_map / 3.0)
anees_gmm = np.mean(all_nees_gmm / 3.0)
anees_gvi = np.mean(all_nees_gvi / 3.0)

# Print results
print(
    f"Estimation Performance over {len(results_map_list)} Trajectories (Concatenated, Non-Normalized NEES)"
)
print(" Method | RMSE (rad) |  RMSE (m)  |  mean(NEES)")
print("-------------------------------------------------")
print(f" MAP    |  {rmse_rad_map:.5f}   |  {rmse_map:.5f}   |  {anees_map:.5f}")
print(f" GMM    |  {rmse_rad_gmm:.5f}   |  {rmse_gmm:.5f}   |  {anees_gmm:.5f}")
print(f" GVI    |  {rmse_rad_gvi:.5f}   |  {rmse_gvi:.5f}   |  {anees_gvi:.5f}")

# Prepare lists to collect results
data = []

print("Estimation Performance per Trajectory")
print("--------------------------------------------------------")
print(" Trial |   Method  | RMSE (rad) | RMSE (m) |  aNEES ")
print("--------------------------------------------------------")

for i, (r_map, r_gmm, r_gvi) in enumerate(
    zip(results_map_list, results_gmm_list, results_gvi_list), start=1
):
    # Compute RMSE (orientation and position)
    rmse_rad_map = np.sqrt(np.mean(np.square(r_map.error[:, 0])))
    rmse_rad_gmm = np.sqrt(np.mean(np.square(r_gmm.error[:, 0])))
    rmse_rad_gvi = np.sqrt(np.mean(np.square(r_gvi.error[:, 0])))

    rmse_map = np.sqrt(np.mean(np.square(r_map.error[:, 1:])))
    rmse_gmm = np.sqrt(np.mean(np.square(r_gmm.error[:, 1:])))
    rmse_gvi = np.sqrt(np.mean(np.square(r_gvi.error[:, 1:])))

    # Mean NEES (non-normalized)
    anees_map = np.mean(r_map.nees / r_map.dof)
    anees_gmm = np.mean(r_gmm.nees / r_gmm.dof)
    anees_gvi = np.mean(r_gvi.nees / r_gvi.dof)

    # Store results for plotting
    data.extend(
        [
            {
                "Trial": i,
                "Method": "MAP (Cauchy)",
                "Metric": "RMSE (rad)",
                "Value": rmse_rad_map,
            },
            {
                "Trial": i,
                "Method": "MAP (GMM)",
                "Metric": "RMSE (rad)",
                "Value": rmse_rad_gmm,
            },
            {
                "Trial": i,
                "Method": "ESGVI",
                "Metric": "RMSE (rad)",
                "Value": rmse_rad_gvi,
            },
            {
                "Trial": i,
                "Method": "MAP (Cauchy)",
                "Metric": "RMSE (m)",
                "Value": rmse_map,
            },
            {
                "Trial": i,
                "Method": "MAP (GMM)",
                "Metric": "RMSE (m)",
                "Value": rmse_gmm,
            },
            {"Trial": i, "Method": "ESGVI", "Metric": "RMSE (m)", "Value": rmse_gvi},
            {
                "Trial": i,
                "Method": "MAP (Cauchy)",
                "Metric": "aNEES",
                "Value": anees_map,
            },
            {"Trial": i, "Method": "MAP (GMM)", "Metric": "aNEES", "Value": anees_gmm},
            {"Trial": i, "Method": "ESGVI", "Metric": "aNEES", "Value": anees_gvi},
        ]
    )

    # Print table rows
    print(
        f"  {i:2d}   |   MAP     | {rmse_rad_map:7.3f}    | {rmse_map:7.3f}  | {anees_map:7.3f}"
    )
    print(
        f"       |   GMM     | {rmse_rad_gmm:7.3f}    | {rmse_gmm:7.3f}  | {anees_gmm:7.3f}"
    )
    print(
        f"       |   GVI     | {rmse_rad_gvi:7.3f}    | {rmse_gvi:7.3f}  | {anees_gvi:7.3f}"
    )
    print("--------------------------------------------------------")


df = pd.DataFrame(data)

# Colors of each method
method_colors = {
    "MAP (Cauchy)": "tab:blue",
    "MAP (GMM)": "tab:purple",
    "ESGVI": "tab:orange",
}

metrics = ["RMSE (rad)", "RMSE (m)", "aNEES"]

# Boxplots
fig, axes = plt.subplots(1, 3, figsize=(12, 4))

for ax, metric in zip(axes, metrics):
    subset = df[df["Metric"] == metric]
    methods = list(method_colors.keys())
    grouped = [subset[subset["Method"] == m]["Value"].values for m in methods]

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
        ax.set_ylabel(r"Orientation RMSE (rad)", fontsize=10)

    elif metric == "RMSE (m)":
        ax.set_ylabel(r"Translation RMSE (m)", fontsize=10)

    else:
        ax.set_ylabel(r"aNEES", fontsize=10)
        # Add horizontal dashed line at y=1 with label
        ax.axhline(
            y=1, color="red", linestyle="--", linewidth=1, label="Expected aNEES"
        )
        confidence_interval = 0.997
        # Add legend only once per subplot
        ci_label = f"${(confidence_interval * 100):.1f}\%$ conf. bounds"
        results_map = results_map_list[4]
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
plt.savefig(f"./figs/{DATASET}/boxplot_{DATASET}.pdf")
plt.show()

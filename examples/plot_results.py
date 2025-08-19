import numpy as np
import scipy.linalg
import navlie as nav
import timeit
import matplotlib.pyplot as plt
import cProfile

from navlie import GaussianResultList, State, Measurement
from typing import List, Tuple
import pickle
import os

SHOW_FIGS = True
SAVE_FIGS = True


# Load data
# TODO: Fix paths for 2D results
with open("./data/results/se2/map_results.pkl", "rb") as f:
    map_data = pickle.load(f)
with open("./data/results/se2/gmm_results.pkl", "rb") as f:
    gmm_data = pickle.load(f)
with open("./data/results/se2/gvi_results.pkl", "rb") as f:
    gvi_data = pickle.load(f)

ground_truth: List[State] = map_data["ground_truth"]
results_map: GaussianResultList = map_data["results_map"]
sim_time: float = map_data["sim_time"]
landmarks = map_data["landmarks"]
results_gmm: GaussianResultList = gmm_data["results_gmm"]
results_gvi: GaussianResultList = gvi_data["results_gvi"]
skew_lambda: float = gvi_data["skew_lambda"]


#####################
##### PLOT GVI ######
#####################
# Plotting parameters
plt.rc("text", usetex=True)
plt.rc("font", family="serif", size=14)
plt.rc("lines", linewidth=2)
plt.rc("axes", grid=True)
plt.rc("grid", linestyle="--")

fig, ax = nav.plot_error(results_map, label="MAP")
ax[0].set_ylabel(r"$\theta$ (rad)")
# Plot GMM
ax[0].plot(
    results_gmm.stamp,
    results_gmm.error[:, 0],
    label="MAP (GMM)",
    linestyle="--",
    color="tab:purple",
)
ax[0].fill_between(
    results_gmm.stamp,
    results_gmm.three_sigma[:, 0],
    -results_gmm.three_sigma[:, 0],
    alpha=0.1,
    color="tab:purple",
)
ax[1].plot(
    results_gmm.stamp,
    results_gmm.error[:, 1],
    label="MAP (GMM)",
    linestyle="--",
    color="tab:purple",
)
ax[1].fill_between(
    results_gmm.stamp,
    results_gmm.three_sigma[:, 1],
    -results_gmm.three_sigma[:, 1],
    alpha=0.1,
    color="tab:purple",
)
ax[2].plot(
    results_gmm.stamp,
    results_gmm.error[:, 2],
    label="MAP (GMM)",
    linestyle="--",
    color="tab:purple",
)
ax[2].fill_between(
    results_gmm.stamp,
    results_gmm.three_sigma[:, 2],
    -results_gmm.three_sigma[:, 2],
    alpha=0.1,
    color="tab:purple",
)
# Plot ESGVI
ax[0].plot(results_gvi.stamp, results_gvi.error[:, 0], label="ESGVI", linestyle="--")
ax[0].fill_between(
    results_gvi.stamp,
    results_gvi.three_sigma[:, 0],
    -results_gvi.three_sigma[:, 0],
    alpha=0.1,
    color="tab:orange",
)
ax[1].set_ylabel(r"$x$ (m)")
ax[1].plot(results_gvi.stamp, results_gvi.error[:, 1], label="ESGVI", linestyle="--")
ax[1].fill_between(
    results_gvi.stamp,
    results_gvi.three_sigma[:, 1],
    -results_gvi.three_sigma[:, 1],
    alpha=0.1,
    color="tab:orange",
)
ax[2].set_ylabel(r"$y$ (m)")
ax[2].plot(results_gvi.stamp, results_gvi.error[:, 2], label="ESGVI", linestyle="--")
ax[2].fill_between(
    results_gvi.stamp,
    results_gvi.three_sigma[:, 2],
    -results_gvi.three_sigma[:, 2],
    alpha=0.1,
    color="tab:orange",
)
ax[2].set_xlabel("Time (s)")
ax[0].legend(loc="upper left", fontsize=7)
ax[1].legend(loc="upper left", fontsize=7)
ax[2].legend(loc="upper left", fontsize=7)
plt.tight_layout()
if SAVE_FIGS:
    plt.savefig(
        f"/home/astirl/Documents/courses/assignments/mech_642/gvi_ws/figs/se2/se2_3sigma_sl_{skew_lambda:.2f}_{sim_time}s.pdf"
    )
if SHOW_FIGS:
    plt.show()

# Poses Plot
fig, ax = nav.plot_poses(
    poses=results_map.state,
    step=100,
    label="MAP (Cauchy)",
    line_color="tab:blue",
    linestyle=":",
)
fig, ax = nav.plot_poses(
    poses=results_gmm.state,
    step=500,
    ax=ax,
    label="MAP (GMM)",
    line_color="tab:purple",
    linestyle="-.",
)
fig, ax = nav.plot_poses(
    poses=results_gvi.state,
    step=100,
    ax=ax,
    label="ESGVI",
    line_color="tab:orange",
    linestyle="--",
)
fig, ax = nav.plot_poses(
    poses=ground_truth, ax=ax, step=None, label="Ground Truth", line_color="tab:green"
)
for l in landmarks:
    ax.plot(l[0], l[1], "x")
ax.set_title("Estimated poses")
ax.set_xlabel(r"$x$ (m)")
ax.set_ylabel(r"$y$ (m)")
ax.legend()
plt.tight_layout()
if SAVE_FIGS:
    plt.savefig(
        f"/home/astirl/Documents/courses/assignments/mech_642/gvi_ws/figs/se2/se2_traj_sl_{skew_lambda:.2f}_{sim_time}s.pdf"
    )
if SHOW_FIGS:
    plt.show()

# Plot NEES
fig, axs = nav.plot_nees(
    results_map, label="MAP (Cauchy)", confidence_interval=None, normalize=True
)

fig, axs = nav.plot_nees(
    results_gmm,
    ax=axs,
    label="MAP (GMM)",
    confidence_interval=None,
    color="tab:purple",
    normalize=True,
)
fig, axs = nav.plot_nees(
    results_gvi, ax=axs, label="ESGVI", confidence_interval=0.997, normalize=True
)

axs.set_xlabel("Time (s)")
axs.set_ylabel(r"Normalized Squared Mahalanobis Distance", fontsize=12)
# axs.set_title("NEES")
if SAVE_FIGS:
    plt.savefig(
        f"/home/astirl/Documents/courses/assignments/mech_642/gvi_ws/figs/se2/se2_NEES_sl_{skew_lambda:.2f}_{sim_time}s.pdf"
    )
if SHOW_FIGS:
    plt.show()

# Comparison table
print("Estimation Performance")
print(" Method | RMSE (rad) |  RMSE (m)  |  aNEES")
print("----------------------------------------")
print(
    f" MAP    |  {np.sqrt(np.mean(np.square(results_map.error[:,0]))):.5f}   |  {np.sqrt(np.mean(np.square(results_map.error[:,1:]))):.5f}   |  {(np.mean(results_map.nees/results_map.dof)):.5f}"
)

print(
    f" GMM    |  {np.sqrt(np.mean(np.square(results_gmm.error[:,0]))):.5f}   |  {np.sqrt(np.mean(np.square(results_gmm.error[:,1:]))):.5f}   |  {(np.mean(results_gmm.nees/results_gmm.dof)):.5f}"
)
print(
    f" ESGVI  |  {np.sqrt(np.mean(np.square(results_gvi.error[:,0]))):.5f}   |  {np.sqrt(np.mean(np.square(results_gvi.error[:,1:]))):.5f}   |  {(np.mean(results_gvi.nees/results_gvi.dof)):.5f}"
)
print(" -------------------------- ")
print(f"Poses: {len(results_gvi.stamp)}")

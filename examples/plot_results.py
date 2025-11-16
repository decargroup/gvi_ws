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

DATASET = "multi"  # "multi"  # "se2_sim"
MEAS_NOISE = "nlos"  # "skew_laplace"
ROT_GT = True
fname = "results.pkl"
save_fname = ""

if DATASET == "se2_sim":
    results_path = os.path.join(DATASET, MEAS_NOISE)

elif DATASET == "cluttered":
    results_path = DATASET

elif DATASET == "multi":
    TRIAL_NUM = 5
    results_path = DATASET
    fname = f"results_{TRIAL_NUM}.pkl"
    save_fname = f"_{TRIAL_NUM}"
else:
    raise NotImplementedError(f"The {DATASET} dataset doesn't exist.")


SHOW_FIGS = True
SAVE_FIGS = True


# Load data
# TODO: Fix paths for 2D results
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


#####################
##### PLOT GVI ######
#####################
# Plotting parameters
plt.rc("text", usetex=True)
plt.rc("font", family="serif", size=14)
plt.rc("lines", linewidth=2)
plt.rc("axes", grid=True)
plt.rc("grid", linestyle="--")

# Override linestyles after plotting
linestyle_map = {
    "MAP (Cauchy)": "-.",
    "MAP (GMM)": "-.",
    "ESGVI": "-.",
}

fig, ax = nav.plot_error(results_map, label="MAP (Cauchy)")
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

for ax_i in ax:
    for line in ax_i.get_lines():
        label = line.get_label()
        if label in linestyle_map:
            line.set_linestyle(linestyle_map[label])

ax[0].legend(loc="upper left", fontsize=10)

plt.tight_layout()
if SAVE_FIGS:
    os.makedirs(f"figs/{results_path}", exist_ok=True)
    plt.savefig(f"./figs/{results_path}/3sigma_sl_{skew_lambda:.2f}{save_fname}.pdf")
    # plt.savefig(f"./figs/{results_path}/3sigma.png")
if SHOW_FIGS:
    plt.show()

# Poses Plot
fig, ax = nav.plot_poses(
    poses=results_map.state,
    step=None,
    label="MAP (Cauchy)",
    line_color="tab:blue",
)
fig, ax = nav.plot_poses(
    poses=results_gmm.state,
    step=None,
    ax=ax,
    label="MAP (GMM)",
    line_color="tab:purple",
)
fig, ax = nav.plot_poses(
    poses=results_gvi.state,
    step=None,
    ax=ax,
    label="ESGVI",
    line_color="tab:orange",
)
fig, ax = nav.plot_poses(
    poses=ground_truth, ax=ax, step=500, label="Ground Truth", line_color="tab:green"
)

for line in ax.get_lines():
    label = line.get_label()
    if label in linestyle_map:
        line.set_linestyle(linestyle_map[label])

for i, l in enumerate(landmarks):
    if i == 0:
        ax.plot(l[0], l[1], "x", color="black", label="Anchor")
    ax.plot(l[0], l[1], "x", color="black")

ax.scatter(
    ground_truth[0].position[0],
    ground_truth[0].position[1],
    facecolors="none",  # no fill
    edgecolors="blue",  # outline color
    marker="o",
    label="Start",
    s=50,
)
ax.scatter(
    ground_truth[-1].position[0],
    ground_truth[-1].position[1],
    facecolors="none",  # no fill
    edgecolors="red",  # outline color
    marker="o",
    label="End",
    s=50,
)

ax.set_title("Estimated poses")
ax.set_xlabel(r"$x$ (m)")
ax.set_ylabel(r"$y$ (m)")
ax.legend(loc="upper right")
plt.tight_layout()
if SAVE_FIGS:
    plt.savefig(f"./figs/{results_path}/traj_sl_{skew_lambda:.2f}{save_fname}.pdf")
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
    plt.savefig(f"./figs/{results_path}/NEES_sl_{skew_lambda:.2f}{save_fname}.pdf")
if SHOW_FIGS:
    plt.show()

if ROT_GT:
    # Plot location of obstacles
    # and rotated ground-truth trajectory
    # Purely for spacing on paper
    R = np.array([[0, -1], [1, 0]])

    def rotate_pose(pose: nav.State, R: np.ndarray):
        new_pose = pose.copy()
        new_pose.position = R @ pose.position
        new_pose.attitude = R @ pose.attitude
        return new_pose

    rotated_poses_gt = [rotate_pose(p, R) for p in ground_truth]
    rotated_poses_map = [rotate_pose(p, R) for p in results_map.state]
    rotated_poses_gmm = [rotate_pose(p, R) for p in results_gmm.state]
    rotated_poses_gvi = [rotate_pose(p, R) for p in results_gvi.state]

    fig, ax = nav.plot_poses(
        poses=rotated_poses_gt, step=1200, label="Ground-Truth", line_color="tab:green"
    )
    fig, ax = nav.plot_poses(
        poses=rotated_poses_map,
        step=None,
        label="MAP (Cauchy)",
        line_color="tab:blue",
        ax=ax,
    )
    fig, ax = nav.plot_poses(
        poses=rotated_poses_gmm,
        step=None,
        label="MAP (GMM)",
        line_color="tab:purple",
        ax=ax,
    )
    fig, ax = nav.plot_poses(
        poses=rotated_poses_gvi,
        step=None,
        label="ESGVI",
        line_color="tab:orange",
        ax=ax,
    )

    ax.scatter(
        rotated_poses_gt[0].position[0],
        rotated_poses_gt[0].position[1],
        facecolors="none",  # no fill
        edgecolors="blue",  # outline color
        marker="o",
        label="Start",
        s=50,
    )
    ax.scatter(
        rotated_poses_gt[-1].position[0],
        rotated_poses_gt[-1].position[1],
        facecolors="none",  # no fill
        edgecolors="red",  # outline color
        marker="o",
        label="End",
        s=50,
    )

    for i, l in enumerate(landmarks):
        if i == 0:
            ax.plot(-l[1], l[0], "x", color="black", label="Anchor")
        ax.plot(-l[1], l[0], "x", color="black")

    ax.set_xlabel(r"$x$ (m)")
    ax.set_ylabel(r"$y$ (m)")

    for line in ax.get_lines():
        label = line.get_label()
        if label in linestyle_map:
            line.set_linestyle(linestyle_map[label])

    if DATASET == "cluttered":
        ax.legend()
        ax.plot([1.3, 1.3], [-1.7, -1.3], color="grey", linewidth=5, label="metal")
        ax.text(1.3, -1.9, "metal", color="black", fontsize=10)

        # Brown line labeled "wood"
        ax.plot([4.4, 4.5], [0.05, 0.3], color="saddlebrown", linewidth=3, label="wood")
        ax.text(4.3, 0.4, "wood", color="black", fontsize=10)

        # Black line labeled "foam"
        ax.plot([-1.1, -1.3], [1.2, 1.5], color="black", linewidth=4, label="foam")
        ax.text(-1.6, 1.55, "foam", color="black", fontsize=10)
        plt.tight_layout()
        ax.set_ylim(-2.0, 2.5)
        ax.set_aspect("equal", adjustable="box")  # force aspect, but respect limits

    if DATASET == "multi":

        ax.plot([-2.35, -2.2], [-1, -1.15], color="grey", linewidth=5)
        ax.text(-2.2, -1.0, "metal", color="black", fontsize=10)

        # Brown line labeled "wood"
        ax.plot([-1.8, -1.7], [1.45, 1.65], color="saddlebrown", linewidth=3)
        ax.text(-1.65, 1.55, "wood", color="black", fontsize=10)

        # Black line labeled "foam"
        ax.plot([5.2, 5.35], [1.35, 1.15], color="black", linewidth=4)
        ax.text(5.0, 1.45, "foam", color="black", fontsize=10)
        plt.tight_layout()
        ax.set_ylim(-2.8, 2.5)
        ax.set_aspect("equal", adjustable="box")  # force aspect, but respect limits
        plt.subplots_adjust(
            left=0.1, bottom=0.02, right=0.975, top=0.975, wspace=0.01, hspace=0.1
        )
        ax.legend(loc="best")

    if SAVE_FIGS:
        plt.savefig(f"figs/{results_path}/traj_sl_{skew_lambda:.2f}{save_fname}.pdf")
        # plt.savefig(f"figs/{DATASET}/{DATASET}_gt_traj.png")

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

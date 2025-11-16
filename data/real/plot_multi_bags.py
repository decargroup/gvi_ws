import numpy as np
import navlie as nav
import matplotlib.pyplot as plt
import pickle
import seaborn as sns
import os

from pymocap.odom import OdomData
from bagpy import bagreader
from navlie.utils.alignment import state_list_to_evo_traj, evo_traj_to_state_list
from navlie.lib.states import SE2State
from evo.core.trajectory import Plane
from gvi_ws.util.load_config import load_config

# Fitting Distributions
from scipy.stats import norm, cauchy
from scipy.integrate import simpson
from scipy.interpolate import interp1d
from gvi_ws.util.fit_skew_laplace import (
    fit_skew_laplace,
    skew_laplace_pdf,
    fit_two_piece_cauchy,
    two_piece_cauchy_pdf,
)
from sklearn.mixture import GaussianMixture

# Typing Stuff
from pymocap import MocapTrajectory, RangeData, Tag, IMUData
from mpl_toolkits.mplot3d import Axes3D
from navlie.types import Measurement
from typing import List

# Plotting parameters
plt.rc("text", usetex=True)
plt.rc("font", family="serif", size=14)
plt.rc("lines", linewidth=2)
plt.rc("axes", grid=True)
plt.rc("grid", linestyle="--")
sns.set_theme(style="whitegrid")

DATASET = "multi"  # TODO: Include more datasets later
IDS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
EXP_BASE_PATH = f"./data/real/{DATASET}"
DATA_BASE_PATH = f"./data/real/bags/{DATASET}"
EXPORT_DATA = True

# Plotting Params
SHOW_PLOTS = False
SAVE_FIGS = True

PLOT_UWB = False

agent = "Husky"
config_path = os.path.join(DATA_BASE_PATH, "anchors.yaml")
config = load_config(config_path)
# This goes from motive software to mocap streaming representation
C_mocap_motive = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])

# Load Tags/Anchors
delays = config["delays"]
anc_ids = config["anc_ids"]
tag_ids = config["tag_ids"]
all_uwb_range_errs = []
np.random.seed(0)

for bag_id in IDS:
    print(f"Starting extraction of {DATASET} dataset with bag_id: {bag_id}")
    bag_path = os.path.join(DATA_BASE_PATH, f"{DATASET}_{bag_id}.bag")

    # EXTRACT DATA
    mocap = MocapTrajectory.from_bag(bag_path, agent)
    uwb = RangeData.from_bag(bag_path, "/uwb/range")
    odom = OdomData.from_bag(bag_path, topic="/husky_velocity_controller/odom")

    # Initial heading/position of husky
    C_mh_0 = mocap.rot_matrix(mocap.stamps)[0]
    t_om_m_0 = mocap.position(mocap.stamps)[0]
    anchors: List[Tag] = []
    tags: List[Tag] = []
    for a_id in anc_ids:
        a_pos = np.array(config[f"anc_{a_id}"]).reshape((-1, 1))
        a_pos = C_mocap_motive @ a_pos
        a = Tag(a_id, parent_id=agent, position=a_pos, antenna_delay=delays[a_id])
        anchors.append(a)

    for t_id in tag_ids:
        t_pos = np.array(config[f"tag_{t_id}"])
        t_pos = (C_mocap_motive @ t_pos) - t_om_m_0
        t_pos = C_mh_0.T @ t_pos
        t = Tag(t_id, parent_id=agent, position=t_pos, antenna_delay=delays[t_id])
        tags.append(t)

    b = bagreader(bag_path)

    # Get portion of trajectory when
    # Husky is moving
    static_mask = mocap.get_static_mask(1.0, 0.001)
    moving_mask = ~static_mask
    indexes = np.arange(len(mocap.stamps))
    moving_stamps = mocap.stamps[moving_mask]
    # Get movement mask for other objects
    interp_mask = interp1d(
        mocap.stamps,
        moving_mask.astype(int),
        kind="nearest",
        bounds_error=False,
        fill_value=0,
    )
    mocap = MocapTrajectory(
        moving_stamps,
        mocap.position(moving_stamps),
        quaternion_data=mocap.quaternion(moving_stamps),
        frame_id=agent,
    )
    range_moving_mask = interp_mask(uwb.stamps).astype(bool)
    odom_moving_mask = interp_mask(odom.stamps).astype(bool)
    # Filter RangeData
    uwb = uwb[range_moving_mask]
    # Filter OdomData
    odom = odom[odom_moving_mask]

    # Calibrate Odometry
    R_oh, t_om_O = odom.calibrate(mocap, compute_pos_offset=True)
    R_ho = R_oh.T
    mocap = mocap.rotate_body_frame(R_ho)
    if SHOW_PLOTS:
        # fig, axs = odom.plot_gyro(mocap)
        fig, axs = odom.plot(mocap=mocap)
        fig.suptitle("Calibrated Body Velocity")
        plt.show()

    for t in tags:
        # Don't adjust position
        t.position = (R_oh @ t.position.T).T  # - t_om_O

    # Update Tag and Anchor positions with calibrated odometry/mocap
    anchor_positions = []
    anc_dict = {}
    for a in anchors:
        anchor_positions.append(a.position)
        anc_dict.update({a.id: a.position.copy()})
    tag_dict = {}
    for t in tags:
        tag_dict.update({t.id: t.position.copy()})

    ## Mocap Trajectory
    ## Poses
    anchor_positions = np.array(anchor_positions)
    poses = mocap.to_navlie(mocap.stamps, pose_type="SE3")
    fig, ax = nav.plot_poses(poses, step=1000, plot_2d=True)
    ax.set_xlabel(r"$x$ (m)")
    ax.set_ylabel(r"$y$ (m)")

    for a in anchors:
        x, y, z = a.position
        if isinstance(ax, Axes3D):
            ax.scatter(x, y, z, label=f"Anchor {a.id}")
            ax.text(x, y, z, f"{a.id}", fontsize=10, color="red")
        else:
            ax.scatter(x, y, label=f"Anchor {a.id}")
            ax.text(x, y, f"{a.id}", fontsize=10, color="red")

    ax.legend()
    fig.suptitle("Raw Mocap Poses")
    if SAVE_FIGS:
        os.makedirs(f"figs/{DATASET}/raw", exist_ok=True)
        fig.savefig(f"figs/{DATASET}/raw/mocap_pose_traj_{bag_id}.pdf")
    if SHOW_PLOTS:
        plt.show()

    ### UWB STUFF ###
    uwb_error = 5.0  # Weird timestamping behaviour, preserves skew
    uwb = uwb.remove_intertag(tags, anchors)
    uwb = uwb.remove_outliers(mocap, tags, max_error=uwb_error, anchors=anchors)
    uwb_calib = uwb.apply_calibration(tags, anchors=anchors)
    fig_calib, axs_calib = uwb_calib.plot(mocap, tags, anchors)
    if SAVE_FIGS:
        fig_calib.savefig(f"figs/{DATASET}/raw/ranges_calibrated_{bag_id}.pdf")

    # Calibrated Plot
    fig_calib_bias, axs_calib_bias, calib_range_errs = uwb_calib.plot_error(
        mocap, tags, anchors, bins=1000, return_bias=True
    )
    axs_calib_bias.set_title("")

    # Add range errors to range errors across all trials
    all_uwb_range_errs.extend(calib_range_errs)

    # Fit distributions
    mu, std = norm.fit(calib_range_errs)
    mu_c, std_c = cauchy.fit(calib_range_errs)
    mu_sl, std_sl, lambda_sl = fit_skew_laplace(calib_range_errs)
    # Mixture Model
    gmm = GaussianMixture(n_components=3, covariance_type="diag", random_state=0)
    gmm.fit(np.array(calib_range_errs).reshape((-1, 1)))

    # X-range for PDFs
    x = np.linspace(-1, 4, 500)
    pdf_gauss = norm.pdf(x, mu, std)
    pdf_cauchy = cauchy.pdf(x, mu_c, std_c)
    pdf_sl = skew_laplace_pdf(x, mu_sl, std_sl, lambda_sl)

    logprob = gmm.score_samples(x.reshape((-1, 1)))
    pdf_gmm = np.exp(logprob)

    # Plot PDFs on ax1
    axs_calib_bias.plot(
        x,
        pdf_gauss,
        "-",
        linewidth=2,
        color="tab:green",
        label=rf"Gaussian Fit\\ $\mu={mu:.2f},\ \sigma={std:.2f}$",
    )
    axs_calib_bias.plot(
        x,
        pdf_cauchy,
        "-.",
        linewidth=2,
        color="tab:blue",
        label=rf"Cauchy Fit\\ $\mu={mu_c:.2f},\ \sigma={std_c:.2f}$",
    )
    axs_calib_bias.plot(
        x,
        pdf_sl,
        "-.",
        linewidth=2,
        color="tab:orange",
        label=rf"Skew-Laplace Fit\\ $\mu={mu_sl:.2f},\ \sigma={std_sl:.2f},\ \lambda={lambda_sl:.3f}$",
    )
    gmm_means = gmm.means_.flatten()
    gmm_stds = np.sqrt(gmm.covariances_.flatten())
    gmm_weights = gmm.weights_
    idx_sort = np.argsort(gmm_weights)[::-1]

    means_str = ", ".join(f"{gmm_means[i]:.2f}" for i in idx_sort)
    stds_str = ", ".join(f"{gmm_stds[i]:.2f}" for i in idx_sort)
    weights_str = ", ".join(f"{gmm_weights[i]:.2f}" for i in idx_sort)

    # Plot GMM PDF
    axs_calib_bias.plot(
        x,
        pdf_gmm,
        "-.",
        linewidth=2,
        color="tab:purple",
        label=rf"Gaussian Mixture Fit\\ $\mu=[{means_str}]$\\ $\sigma=[{stds_str}]$\\ $w=[{weights_str}]$",
    )

    # Axis Labels
    axs_calib_bias.set_xlabel(rf"Range Error, $e_r$ (m)", fontsize=14)
    axs_calib_bias.set_ylabel("Probability Density", fontsize=13)
    axs_calib_bias.set_xlim(left=-0.5, right=3)

    # Tick label font size
    axs_calib_bias.tick_params(axis="both", labelsize=10)

    # Legend
    axs_calib_bias.legend(fontsize=14, loc="upper right", frameon=True)
    fig_calib_bias.tight_layout()
    if SAVE_FIGS:
        fig_calib_bias.savefig(f"figs/{DATASET}/raw/range_errors_{bag_id}.pdf")

    if SHOW_PLOTS:
        plt.show()

    if EXPORT_DATA:
        input_data = odom.to_navlie(use_ros_stamps=False)
        # Project the ranges to 2D
        uwb.compute_projected_range(mocap, anchors, tags)
        meas_data = uwb.to_navlie(
            tags=tags,
            anchors=anchors,
            use_ros_stamps=False,
            variance=std_sl**2,
            use_2d=True,
        )
        uwb_calib.compute_projected_range(mocap, anchors, tags)
        meas_data_calib = uwb_calib.to_navlie(
            tags=tags,
            anchors=anchors,
            use_ros_stamps=False,
            use_2d=True,
        )
        # Project the 3D Pose to 2D
        ground_truth_se3 = mocap.to_navlie(use_ros_stamps=False, pose_type="SE3")
        ground_truth_pose = state_list_to_evo_traj(ground_truth_se3)
        ground_truth_pose.project(Plane.XY)
        ground_truth_pose.positions_xyz
        ground_truth_pose.orientations_quat_wxyz
        ground_truth_se2 = evo_traj_to_state_list(ground_truth_pose)
        ground_truth_proj = [
            SE2State(
                value=np.block(
                    [
                        [x.attitude[0:2, 0:2], x.position[0:2].reshape((2, 1))],
                        [np.zeros((1, 2)), np.ones((1, 1))],
                    ]
                ),
                stamp=x.stamp,
                state_id=x.state_id,
            )
            for x in ground_truth_se2
        ]

        os.makedirs(EXP_BASE_PATH, exist_ok=True)

        save_path = os.path.join(EXP_BASE_PATH, f"exported_data_{bag_id}.pkl")
        with open(save_path, "wb") as f:
            pickle.dump(
                {
                    "ground_truth": ground_truth_proj,
                    "input_data": input_data,
                    "meas_data": meas_data,
                    "meas_data_calib": meas_data_calib,
                    "anchors": anc_dict,
                    "tags": tag_dict,
                },
                f,
            )
# Fit Distributions using all UWB Data
all_uwb_range_errs = np.array(all_uwb_range_errs).flatten()
print(f"Number of range measurements: {len(all_uwb_range_errs)}")
# Plot histogram
fig, ax_b = plt.subplots()
ax_b.hist(
    all_uwb_range_errs,
    bins="fd",
    alpha=0.6,
    color="grey",
    edgecolor="black",
    density=True,
)
ax_b.set_title("")
ax_b.set_xlabel(r"Range Error, $e_r$ (m)")
ax_b.set_ylabel(r"Probability Density")
# Fit distributions
mu, std = norm.fit(all_uwb_range_errs)
mu_c, std_c = cauchy.fit(all_uwb_range_errs)
mu_c_mod, c_pos, c_neg = fit_two_piece_cauchy(all_uwb_range_errs)
mu_sl, std_sl, lambda_sl = fit_skew_laplace(all_uwb_range_errs)
# Mixture Model
gmm = GaussianMixture(n_components=3, covariance_type="diag", random_state=0)
gmm.fit(np.array(all_uwb_range_errs).reshape((-1, 1)))
# X-range for PDFs
x = np.linspace(-1, 4, 500)
pdf_gauss = norm.pdf(x, mu, std)
pdf_cauchy = cauchy.pdf(x, mu_c, std_c)
pdf_c_mod = two_piece_cauchy_pdf(x, mu_c_mod, c_pos, c_neg)

pdf_sl = skew_laplace_pdf(x, mu_sl, std_sl, lambda_sl)
logprob = gmm.score_samples(x.reshape((-1, 1)))
pdf_gmm = np.exp(logprob)

# ax_b.plot(
#     x,
#     pdf_gauss,
#     "-",
#     linewidth=2,
#     color="tab:green",
#     label=rf"Gaussian Fit\\ $\mu={mu:.2f},\ \sigma={std:.2f}$",
# )
# ax_b.plot(
#     x,
#     pdf_cauchy,
#     "-.",
#     linewidth=2,
#     color="tab:blue",
#     label=rf"Cauchy Fit\\ $\mu={mu_c:.2f},\ \sigma={std_c:.2f}$",
# )
ax_b.plot(
    x,
    pdf_c_mod,
    "-.",
    linewidth=2,
    color="tab:blue",
    label=rf"Asymmetric Cauchy Fit\\ $\mu={mu_c:.2f}$,\ $c^-={c_neg:.2f}$,\ $c^+={c_pos:.2f}$",
)
ax_b.plot(
    x,
    pdf_sl,
    "-.",
    linewidth=2,
    color="tab:orange",
    label=rf"Skew-Laplace Fit\\ $\mu={mu_sl:.2f},\ \sigma={std_sl:.2f},\ \lambda={lambda_sl:.3f}$",
)
gmm_means = gmm.means_.flatten()
gmm_stds = np.sqrt(gmm.covariances_.flatten())
gmm_weights = gmm.weights_
idx_sort = np.argsort(gmm_weights)[::-1]

means_str = ", ".join(f"{gmm_means[i]:.2f}" for i in idx_sort)
stds_str = ", ".join(f"{gmm_stds[i]:.2f}" for i in idx_sort)
weights_str = ", ".join(f"{gmm_weights[i]:.2f}" for i in idx_sort)

# Plot GMM PDF
ax_b.plot(
    x,
    pdf_gmm,
    "-.",
    linewidth=2,
    color="tab:purple",
    label=rf"Gaussian Mixture Fit\\ $\mu=[{means_str}]$\\ $\sigma=[{stds_str}]$\\ $w=[{weights_str}]$",
)

# Axis Labels
ax_b.set_xlabel(rf"Range Error, $e_r$ (m)", fontsize=14)
ax_b.set_ylabel("Probability Density", fontsize=13)
ax_b.set_xlim(left=-0.5, right=1.4)

# Tick label font size
ax_b.tick_params(axis="both", labelsize=10)

# Legend
ax_b.legend(fontsize=14, loc="upper right", frameon=True)
fig.tight_layout()

if SAVE_FIGS:
    fig.savefig(f"figs/{DATASET}/raw/all_trials_histogram.pdf")

plt.show()

noise_params = {
    "Gaussian": [mu, std],
    "Cauchy": [mu_c, std_c],
    "GMM": [
        gmm.means_.flatten().tolist(),
        np.sqrt(gmm.covariances_.flatten().tolist()),
        gmm.weights_.tolist(),
    ],
    "Skew Laplace": [mu_sl, std_sl, lambda_sl],
    "Asymmetric Cauchy": [mu_c_mod, c_pos, c_neg],
}
if EXPORT_DATA:
    save_path_uwb = os.path.join(EXP_BASE_PATH, f"uwb_data.pkl")
    with open(save_path_uwb, "wb") as f:
        pickle.dump(
            {
                "total_range_errs": all_uwb_range_errs,
                "fitted_noise_params": noise_params,
            },
            f,
        )

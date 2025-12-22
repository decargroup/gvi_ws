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
from gvi_ws.util.fit_skew_laplace import fit_skew_laplace, skew_laplace_pdf
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

# TODO: Include more datasets later
DATASET = "multi"  # "cluttered" # "multi"
ID = 9
EXP_PATH = f"./data/real/{DATASET}/exported_data_{ID}.pkl"
EXPORT_DATA = False

# Plotting Params
SAVE_FIGS = False
PLOT_ODOM = True
PLOT_MOCAP = False
PLOT_TAGS_ANCHORS = False
PLOT_UWB = True


bag_path = f"data/real/bags/{DATASET}/{DATASET}_{ID}.bag"
if DATASET == "cluttered":
    # cluttered dataset only has 1 trajectory
    bag_path = f"data/real/bags/{DATASET}/{DATASET}.bag"
config_path = f"data/real/bags/{DATASET}/anchors.yaml"
agent = "Husky"
config = load_config(config_path)
# This goes from motive software to mocap streaming representation
C_mocap_motive = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])

# EXTRACT DATA
mocap = MocapTrajectory.from_bag(bag_path, agent)
uwb = RangeData.from_bag(bag_path, "/uwb/range")
# odom = OdomData.from_bag(bag_path, topic="/odometry/filtered")
odom = OdomData.from_bag(bag_path, topic="/husky_velocity_controller/odom")

# LOAD TAGS/ANCHORS
delays = config["delays"]
anc_ids = config["anc_ids"]
a_pos_motive = []
tag_ids = config["tag_ids"]
t_pos_motive = []

# Initial heading/position of husky
C_mh_0 = mocap.rot_matrix(mocap.stamps)[0]
t_om_m_0 = mocap.position(mocap.stamps)[0]
anchors: List[Tag] = []
tags: List[Tag] = []
for a_id in anc_ids:
    a_pos = np.array(config[f"anc_{a_id}"]).reshape((-1, 1))
    a_pos_motive.append(a_pos)
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
print(b.topic_table)

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

if PLOT_ODOM:
    fig, axs = odom.plot_gyro(mocap)
    fig, axs = odom.plot(mocap=mocap)
    fig.suptitle("Uncalibrated Body Velocity")
    if SAVE_FIGS:
        fig.savefig(f"figs/{DATASET}/raw/uncalib_odometry.pdf")
    plt.show()

# Calibrate Odometry
R_oh, t_om_O = odom.calibrate(mocap, compute_pos_offset=True)
R_ho = R_oh.T
mocap = mocap.rotate_body_frame(R_ho)

t_translation = np.array([0.32536226, -0.30953877, 0.00398481])
for t in tags:
    t.position = (R_oh @ t.position.T).T  # - t_om_O

# Replot Calibrated Odometry
if PLOT_ODOM:
    fig, axs = odom.plot_gyro(mocap)
    fig, axs = odom.plot(mocap=mocap)
    fig.suptitle("Calibrated Body Velocity")
    if SAVE_FIGS:
        fig.savefig(f"figs/{DATASET}/raw/calib_odometry.pdf")
    plt.show()

# Update Tag and Anchor positions with calibrated odometry/mocap
anchor_positions = []
anc_dict = {}
for a in anchors:
    anchor_positions.append(a.position)
    anc_dict.update({a.id: a.position.copy()})
tag_dict = {}
for t in tags:
    tag_dict.update({t.id: t.position.copy()})

if PLOT_TAGS_ANCHORS:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    # Initial attitude/position of husky
    C_mo_0: np.ndarray = mocap.rot_matrix(mocap.stamps)[0]
    t_om_m_0 = mocap.position(mocap.stamps)[0]

    # Odom
    ax.scatter(t_om_m_0[0], t_om_m_0[1], t_om_m_0[2], label="Odom")
    ax.text(t_om_m_0[0], t_om_m_0[1], t_om_m_0[2], "Odom")

    # Tags
    for t_id, t_pos in tag_dict.items():
        r_tm_M = (C_mo_0 @ t_pos) + t_om_m_0
        ax.scatter(r_tm_M[0], r_tm_M[1], r_tm_M[2])
        ax.text(r_tm_M[0], r_tm_M[1], r_tm_M[2], str(t_id))

    # Anchors
    for a_id, a_pos in anc_dict.items():
        ax.scatter(a_pos[0], a_pos[1], a_pos[2])
        ax.text(a_pos[0], a_pos[1], a_pos[2], str(a_id))

    ax.legend()
    fig.tight_layout()
    if SAVE_FIGS:
        fig.savefig(f"figs/{DATASET}/raw/husky_odom_mocap_offset.pdf")
    plt.show()

### POSITION
if PLOT_MOCAP:
    fig, axs = plt.subplots(3, 1, sharex=True)
    axs: List[plt.Axes] = axs
    pos = mocap.position(mocap.stamps)
    axs[0].plot(mocap.plotting_stamps, pos[:, 0])
    axs[0].set_ylabel(r"$x$ (m)")
    axs[1].plot(mocap.plotting_stamps, pos[:, 1])
    axs[1].set_ylabel(r"$y$ (m)")
    axs[2].plot(mocap.plotting_stamps, pos[:, 2])
    axs[2].set_ylabel(r"$z$ (m)")
    is_static = mocap.is_static(mocap.stamps)
    axs[2].plot(mocap.plotting_stamps, is_static.astype(int), label="Static")
    axs[2].set_xlabel("Time (s)")
    axs[0].set_title("Mocap Position Trajectory")
    axs[2].legend()
    fig.tight_layout()
    if SAVE_FIGS:
        fig.savefig(f"figs/{DATASET}/raw/mocap_position_traj.pdf")

    ### VELOCITY
    fig, axs = plt.subplots(3, 1, sharex=True, sharey=True)
    # fig.tight_layout()
    axs: List[plt.Axes] = axs
    vel = mocap.velocity(mocap.stamps)
    axs[0].plot(mocap.plotting_stamps, vel[:, 0])
    axs[0].set_ylabel(r"$\dot{x}$ (m/s)")
    axs[1].plot(mocap.plotting_stamps, vel[:, 1])
    axs[1].set_ylabel(r"$\dot{y}$ (m/s)")
    axs[2].plot(mocap.plotting_stamps, vel[:, 2])
    axs[2].set_ylabel(r"$\dot{z}$ (m/s)")
    axs[0].set_title("Mocap Velocity Trajectory")
    fig.tight_layout()

    ### ANGULAR VELOCITY
    fig, axs = plt.subplots(3, 1, sharex=True, sharey=True)
    axs: List[plt.Axes] = axs
    omega = mocap.angular_velocity(mocap.stamps)
    axs[0].plot(mocap.stamps, omega[:, 0])
    axs[1].plot(mocap.stamps, omega[:, 1])
    axs[2].plot(mocap.stamps, omega[:, 2])
    axs[0].set_title("Mocap Angular Velocity Trajectory")
    axs[0].set_ylim(-1.05, 1.05)

    ### POSES
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
        fig.savefig(f"figs/{DATASET}/raw/mocap_pose_traj.pdf")

    plt.show()

### UWB STUFF ###
uwb_error = 5.0  # For weird timestamping behaviour while preserving general skew
uwb = uwb.remove_intertag(tags, anchors)
uwb = uwb.remove_outliers(mocap, tags, max_error=uwb_error, anchors=anchors)
uwb_calib = uwb.apply_calibration(tags, anchors=anchors)

# UWB Biases
if PLOT_UWB:
    # Range Plots
    fig, axs = uwb.plot(mocap, tags, anchors)
    fig_calib, axs_calib = uwb_calib.plot(mocap, tags, anchors)
    fig_calib.suptitle("Calibrated Range Data")
    if SAVE_FIGS:
        fig.savefig(f"figs/{DATASET}/raw/ranges.pdf")
        fig_calib.savefig(f"figs/{DATASET}/raw/ranges_calibrated.pdf")

    # Bias Plots
    fig_bias, axs_bias, biases = uwb.plot_error(
        mocap, tags, anchors, bins=1000, return_bias=True
    )
    fig_bias.suptitle(r"\textbf{Uncalibrated UWB}")

    # Calibrated Plot
    fig_calib_bias, axs_calib_bias, calib_biases = uwb_calib.plot_error(
        mocap, tags, anchors, bins=1000, return_bias=True
    )
    axs_calib_bias.set_title("")

    biases_list = [biases, calib_biases]
    axs_list = [axs_bias, axs_calib_bias]
    for i, b in enumerate(biases_list):
        ax_b = axs_list[i]
        # Fit distributions
        mu, std = norm.fit(b)
        mu_c, std_c = cauchy.fit(b)
        mu_sl, std_sl, lambda_sl = fit_skew_laplace(b)
        # Mixture Model
        gmm = GaussianMixture(n_components=3, covariance_type="diag", random_state=0)
        gmm.fit(np.array(b).reshape((-1, 1)))

        # X-range for PDFs
        x = np.linspace(0 - 1, 0 + 3, 500)
        pdf_gauss = norm.pdf(x, mu, std)
        pdf_cauchy = cauchy.pdf(x, mu_c, std_c)
        pdf_sl = skew_laplace_pdf(x, mu_sl, std_sl, lambda_sl)

        logprob = gmm.score_samples(x.reshape((-1, 1)))
        pdf_gmm = np.exp(logprob)

        # Plot PDFs on ax1
        ax_b.plot(
            x,
            pdf_gauss,
            "-",
            linewidth=2,
            color="tab:blue",
            label=rf"Gaussian Fit\\ $\mu={mu:.2f},\ \sigma={std:.2f}$",
        )
        # ax_b.plot(
        #     x,
        #     pdf_cauchy,
        #     ":",
        #     linewidth=2,
        #     color="tab:orange",
        #     label=f"Cauchy Fit\n$\mu$={mu_c:.2f}, σ={std_c:.2f}",
        # )
        ax_b.plot(
            x,
            pdf_sl,
            "--",
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
        ax_b.set_xlim(left=-0.5, right=3)

        # Tick label font size
        ax_b.tick_params(axis="both", labelsize=10)

        # Legend
        ax_b.legend(fontsize=14, loc="upper right", frameon=True)
        fig_bias.tight_layout()
        fig_calib_bias.tight_layout()
    if SAVE_FIGS:

        fig_bias.savefig(f"figs/{DATASET}/raw/uwb_range_error.pdf")
        fig_calib_bias.savefig(f"figs/{DATASET}/raw/uwb_range_error_calibrated.pdf")
        fig_calib_bias.savefig(f"figs/{DATASET}/raw/uwb_range_error_calibrated.png")
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

    os.makedirs(EXP_PATH, exist_ok=True)

    save_path = os.path.join(EXP_PATH, "exported_data.pkl")
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

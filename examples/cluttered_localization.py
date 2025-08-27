# %%
import numpy as np
import navlie as nav
import matplotlib.pyplot as plt
import pickle
import os

from gvi_ws.graph.losses import SkewLaplaceLoss, GaussianLoss, StudentTLoss, CauchyLoss
from gvi_ws.graph.construct_esgvi import generate_esgvi_graph, esgvi_from_map
from gvi_ws.util.data_generation import DataGenerator
from gvi_ws.util.map_batch import construct_planar_map, construct_gmm_map
from gvi_ws.util.load_config import load_config
from navlie.types import State, StateWithCovariance, Measurement, Input
from navlie.lib.states import MatrixLieGroupState, SE2State, SE3State
from navlie.filters import generate_sigmapoints
from navlie.lib.models import (
    BodyFrameVelocity,
)
from navlie.batch import losses

from typing import List, Dict


# Params
est_params = load_config("./config/cluttered_localization.yaml")

MAX_TIME = est_params["MAX_TIME"]  # s
DATA_PATH = est_params["DATA_PATH"]
USE_CALIB = est_params["use_calibrated_uwb"]
WARMSTART = est_params["warmstart"]

# Choose which estimator
USE_MAP = est_params["USE_MAP"]
USE_GVI = est_params["USE_GVI"]
USE_GMM = est_params["USE_GMM"]

# GMM Params
gmm_means = est_params["gmm_means"]
gmm_std_devs = est_params["gmm_std_devs"]
gmm_covariances = np.square(gmm_std_devs)
gmm_weights = est_params["gmm_weights"]


# Exporting Params
EXPORT_DATA = est_params["EXPORT_DATA"]
EXP_PATH = est_params["EXP_PATH"]

# Data Params
meas_std_dev_map = float(est_params["meas_std_dev_map"])
meas_mean_map = float(est_params["meas_mean_map"])

meas_std_dev_gvi = float(est_params["meas_std_dev_gvi"])
meas_mean_gvi = float(est_params["meas_mean_gvi"])


# %%
# Load data
with open(DATA_PATH, "rb") as f:
    data = pickle.load(f)

# Access individual variables
input_data_raw: List[Input] = data["input_data"]
input_data = [u for u in input_data_raw if u.stamp <= MAX_TIME]
input_stamps = [u.stamp for u in input_data]
if USE_CALIB:
    meas_data_map: List[Measurement] = data["meas_data_calib"]
    meas_data_gvi: List[Measurement] = data["meas_data_calib"]
else:
    meas_data_map: List[Measurement] = data["meas_data"]
    meas_data_gvi: List[Measurement] = data["meas_data"]
meas_stamps = [meas.stamp for meas in meas_data_map]
gt_data: List[SE2State] = data["ground_truth"]
gt_stamps = [gt.stamp for gt in gt_data]
anchors: Dict[str, np.ndarray] = data["anchors"]

match_inputs_meas = nav.associate_stamps(
    input_stamps, meas_stamps, max_difference=0.005
)
inputs = []
measurements_map = []
measurements_gvi = []
for match in match_inputs_meas:
    inputs.append(input_data[match[0]])
    meas_gvi = meas_data_gvi[match[1]]
    meas_map = meas_data_map[match[1]]
    if not USE_CALIB:
        meas_map.model._R = np.array([meas_std_dev_map**2])
        meas_gvi.model._R = np.array([meas_std_dev_gvi**2])

    measurements_map.append(meas_map)
    measurements_gvi.append(meas_gvi)
match_inputs_gt = nav.associate_stamps(input_stamps, gt_stamps, max_difference=0.01)
ground_truth = []
for match in match_inputs_gt:
    ground_truth.append(gt_data[match[1]])

# %%
# Initialization and Models
np.random.seed(1)
x0: SE2State = ground_truth[0]
P0 = np.identity(3) * float(est_params["P0_init"])
x0_state_map = x0.plus(nav.randvec(P0))
x0_state_gvi = x0_state_map.copy()
# x0_state = x0.copy()
Q_d: List[float] = est_params["Q_proc"]
Q_d[0] = float(Q_d[0])
Q_d[1] = float(Q_d[1])
Q_d[2] = float(Q_d[2])
Q_d = np.diag(Q_d)
process_model = BodyFrameVelocity(Q_d)

# Losses
if est_params["map_loss"] == "cauchy":
    map_loss_fun = losses.CauchyLoss()
else:
    map_loss_fun = losses.L2Loss()

if est_params["gvi_loss"] == "skew_laplace":
    skew_lambda = est_params["skew_lambda"]
    gvi_loss_fun = SkewLaplaceLoss(lamb=skew_lambda)

elif est_params["gvi_loss"] == "cauchy":
    gvi_loss_fun = CauchyLoss()

else:
    gvi_loss_fun = GaussianLoss()

# %%
# MAP Estimation
print("Starting MAP Estimation")
problem, init_pose_est = construct_planar_map(
    x0=x0_state_map.copy(),
    P0=P0.copy(),
    input_data=inputs,
    process_model=process_model,
    meas_data=measurements_map,
    loss_fun=map_loss_fun,
    slam=False,
    step_tol=float(est_params["tolerance"]),
)
# Initialize ESGVI information
problem.variables = {k: v.copy() for k, v in problem.variables_init.items()}
problem._compute_size_of_problem()
_, H, _ = problem.compute_error_jac_cost()
esgvi_init_info: np.ndarray = (H.T @ H).copy()

if est_params["USE_MAP"]:
    # Solve MAP
    opt_results = problem.solve()
    variables_opt = opt_results["variables"]
    estimate_list_map: List[nav.types.StateWithCovariance] = []
    pose_list_map: List[SE2State] = []
    for pose in init_pose_est:
        state = variables_opt[pose.state_id]
        # Extract the covariance for only this current pose state
        cov = problem.get_covariance_block(pose.state_id, pose.state_id)
        estimate_list_map.append(StateWithCovariance(state, cov))
        pose_list_map.append(state)

    estimate_stamps_map = [float(x.state.stamp) for x in estimate_list_map]
    gt_stamps = [x.stamp for x in ground_truth]
    matches = nav.associate_stamps(estimate_stamps_map, gt_stamps)

    est_list_map = []
    gt_list = []
    for match in matches:
        gt_list.append(ground_truth[match[1]])
        est_list_map.append(estimate_list_map[match[0]])

    results_map = nav.GaussianResultList.from_estimates(est_list_map, gt_list)
    fig, ax = nav.plot_error(results_map, label="MAP (Cauchy)")
    plt.show()
    fig, ax = nav.plot_poses(poses=results_map.state, step=100, label="MAP (Cauchy)")
    fig, ax = nav.plot_poses(
        poses=results_map.state_true, ax=ax, step=None, label="Ground Truth"
    )
    ax.legend()
    plt.tight_layout()
    plt.show()
    if EXPORT_DATA:
        path = "./data/results/cluttered"
        os.makedirs(path, exist_ok=True)
        data_export = os.path.join(path, "map_results.pkl")
        with open(data_export, "wb") as f:
            pickle.dump(
                {
                    "results_map": results_map,
                    "sim_time": MAX_TIME,
                    "ground_truth": gt_data,
                    "landmarks": list(anchors.values()),
                },
                f,
            )

if USE_GMM:
    print("Starting MAP GMM Estimation")
    problem_gmm, init_pose_est_gmm = construct_gmm_map(
        x0=x0_state_map.copy(),
        P0=P0.copy(),
        input_data=inputs,
        process_model=process_model,
        meas_data=measurements_map,
        means=gmm_means,
        covariances=gmm_covariances,
        weights=gmm_weights,
        step_tol=float(est_params["tolerance"]),
    )
    opt_results_gmm = problem_gmm.solve()
    variables_opt_gmm = opt_results_gmm["variables"]
    estimate_list_gmm: List[nav.types.StateWithCovariance] = []
    pose_list_gmm: List[SE2State] = []
    for pose in init_pose_est_gmm:
        state = variables_opt_gmm[pose.state_id]
        # Extract the covariance for only this current pose state
        cov = problem.get_covariance_block(pose.state_id, pose.state_id)
        estimate_list_gmm.append(StateWithCovariance(state, cov))
        pose_list_gmm.append(state)

    estimate_stamps_gmm = [float(x.state.stamp) for x in estimate_list_gmm]
    gt_stamps = [x.stamp for x in ground_truth]
    matches = nav.associate_stamps(estimate_stamps_gmm, gt_stamps)

    est_list_gmm = []
    gt_list = []
    for match in matches:
        gt_list.append(ground_truth[match[1]])
        est_list_gmm.append(estimate_list_gmm[match[0]])

    results_gmm = nav.GaussianResultList.from_estimates(est_list_gmm, gt_list)
    fig, ax = nav.plot_error(results_gmm, label="MAP (GMM)")
    plt.show()
    fig, ax = nav.plot_poses(poses=results_gmm.state, step=100, label="MAP (GMM)")
    fig, ax = nav.plot_poses(
        poses=results_gmm.state_true, ax=ax, step=None, label="Ground Truth"
    )
    ax.legend()
    plt.tight_layout()
    plt.show()
    if EXPORT_DATA:
        path = "./data/results/cluttered"
        os.makedirs(path, exist_ok=True)
        data_export = os.path.join(path, "gmm_results.pkl")
        with open(data_export, "wb") as f:
            pickle.dump(
                {
                    "results_gmm": results_gmm,
                    "sim_time": MAX_TIME,
                    "ground_truth": gt_data,
                    "landmarks": list(anchors.values()),
                },
                f,
            )

# %%
# GVI Solving
if USE_GVI:
    print("Building ESGVI Factor Graph...")
    if WARMSTART:
        esgvi_graph = esgvi_from_map(
            map_problem=problem,
            proc_cubature="gh",
            meas_cubature="gh",
            cubature_order=3,
            proc_loss=GaussianLoss(),
            meas_loss=gvi_loss_fun,
        )
    else:
        esgvi_graph = generate_esgvi_graph(
            x0_state_gvi.copy(),
            P0=P0.copy(),
            init_info_matrix=esgvi_init_info,
            input_data=inputs,
            meas_data=measurements_gvi,
            process_model=process_model,
            proc_cubature="gh",
            meas_cubature="gh",
            cubature_order=3,
            meas_loss=gvi_loss_fun,
            proc_loss=GaussianLoss(),
        )
    # Set ESGVI Params
    esgvi_graph.verbose = est_params["verbose"]
    esgvi_graph.max_iters = est_params["gvi_iters"]
    esgvi_graph.backtrack_iters = est_params["gvi_backtrack"]
    esgvi_graph.init_step_distance = float(est_params["gvi_init_step_dist"])
    esgvi_graph.step_tol = float(est_params["tolerance"])
    esgvi_graph.backtrack_multiplier = float(est_params["backtrack_multiplier"])

    # %%
    # Start solving
    gvi_states_solved = esgvi_graph.solve()
    est_list_gvi_raw: List[StateWithCovariance] = []
    pose_list_gvi: List[SE2State] = []
    for state_id, state in gvi_states_solved.items():
        cov = esgvi_graph.get_covariance_block(state_id, state_id)
        pose_list_gvi.append(state)
        est_list_gvi_raw.append(StateWithCovariance(state, cov))

    est_stamps_gvi = [float(x.state.stamp) for x in est_list_gvi_raw]
    gt_stamps = [float(x.stamp) for x in ground_truth]
    matches = nav.associate_stamps(est_stamps_gvi, gt_stamps)
    est_list_gvi: List[StateWithCovariance] = []
    gt_list = []
    for match in matches:
        est_list_gvi.append(est_list_gvi_raw[match[0]])
        gt_list.append(ground_truth[match[1]])

    results_gvi = nav.GaussianResultList.from_estimates(est_list_gvi, gt_list)

    fig, ax = nav.plot_error(results_gvi, label="ESGVI", color="tab:orange")
    plt.show()
    fig, ax = nav.plot_poses(
        poses=results_gvi.state, step=100, label="ESGVI", line_color="tab:orange"
    )
    fig, ax = nav.plot_poses(
        poses=results_gvi.state_true,
        ax=ax,
        step=None,
        label="Ground Truth",
        line_color="tab:green",
    )
    ax.legend()
    plt.tight_layout()
    plt.show()

    if EXPORT_DATA:
        path = "./data/results/cluttered"
        os.makedirs(path, exist_ok=True)
        data_export = os.path.join(path, "gvi_results.pkl")
        with open(data_export, "wb") as f:
            pickle.dump(
                {
                    "results_gvi": results_gvi,
                    "skew_lambda": est_params["skew_lambda"],
                    "sim_time": MAX_TIME,
                    "ground_truth": gt_data,
                    "landmarks": list(anchors.values()),
                },
                f,
            )
        print("Exported ESGVI results.")
# print("Estimation Performance")
# print(" Method | RMSE (rad) |  RMSE (m)  |  aNEES")
# print("----------------------------------------")
# print(
#     f" ESGVI  |  {np.sqrt(np.mean(np.square(results_gvi.error[:,0]))):.5f}   |  {np.sqrt(np.mean(np.square(results_gvi.error[:,1:]))):.5f}   |  {(np.mean(results_gvi.nees/results_gvi.dof)):.5f}"
# )
# print(
#     f" MAP    |  {np.sqrt(np.mean(np.square(results_map.error[:,0]))):.5f}   |  {np.sqrt(np.mean(np.square(results_map.error[:,1:]))):.5f}   |  {(np.mean(results_map.nees/results_map.dof)):.5f}"
# )
# print(
#     f" GMM    |  {np.sqrt(np.mean(np.square(results_gmm.error[:,0]))):.5f}   |  {np.sqrt(np.mean(np.square(results_gmm.error[:,1:]))):.5f}   |  {(np.mean(results_gmm.nees/results_gmm.dof)):.5f}"
# )
# print(" -------------------------- ")
# print(f"Total degrees of freedom x: {esgvi_graph._graph_total_dof}")
# print(f"Poses: {(esgvi_graph._num_poses):.0f}")

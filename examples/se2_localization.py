# %%
import numpy as np
import scipy.linalg
import navlie as nav
import timeit
import matplotlib.pyplot as plt
import cProfile

from gvi_ws.graph.losses import SkewLaplaceLoss, GaussianLoss, StudentTLoss, CauchyLoss
from gvi_ws.graph.construct_esgvi import generate_esgvi_graph, esgvi_from_map
from gvi_ws.util.data_generation import DataGenerator
from gvi_ws.util.map_batch import construct_planar_map
from gvi_ws.util.load_config import load_config
from navlie.types import State, StateWithCovariance, Measurement, Input
from navlie.lib.states import MatrixLieGroupState, SE2State, VectorState
from navlie.filters import generate_sigmapoints
from navlie.lib.models import (
    BodyFrameVelocity,
    RangePointToAnchor,
    PointRelativePosition,
    RangePoseToAnchor,
)
from navlie.batch.losses import CauchyLoss, L2Loss

from typing import List, Tuple
import pickle
import os

# %%
if __name__ == "__main__":
    config = load_config("config/se2_localization.yaml")
    noise_config = load_config("config/noise_config.yaml")

    np.random.seed(config["SEED"])

    T_END = config["T_END"]
    NOISE = config["NOISE"]
    USE_FIT = config["USE_FIT"]
    CUB_METHOD_PROC = config["SP_METHOD_PROC"]
    CUB_METHOD_MEAS = config["SP_METHOD_MEAS"]
    CUB_ORDER = config["CUB_ORDER"]
    MAP_INIT = config["MAP_INIT"]
    TIME_IT = config["TIME_IT"]

    VERBOSE = config["VERBOSE"]
    MAX_ITERS = config["MAX_ITERS"]
    BACK_ITERS = config["BACK_ITERS"]
    INIT_STEP_SIZE = float(config["INIT_STEP_SIZE"])

    SAVE_FIGS = config["SAVE_FIGS"]
    SHOW_FIGS = config["SHOW_FIGS"]
    STEP_TOL = float(config["STEP_TOL"])

    # Noise Params
    PROC_NOISE = noise_config["PROC_NOISE"]
    MEAS_NOISE = noise_config["MEAS_NOISE"]
    MAP_LOSS_FUN = noise_config["MAP_LOSS_FUN"]
    GVI_LOSS_FUN = noise_config["GVI_LOSS_FUN"]
    if MAP_LOSS_FUN == "cauchy":
        MAP_LOSS_FUN = CauchyLoss()
    else:
        MAP_LOSS_FUN = L2Loss()
    if GVI_LOSS_FUN == "skew_laplace":
        gvi_skew_lambda = float(noise_config["GVI_SKEW_LAMBDA"])
        GVI_LOSS_FUN = SkewLaplaceLoss(lamb=gvi_skew_lambda)
    elif GVI_LOSS_FUN == "student_t":
        gvi_t_dof = float(noise_config["GVI_T_DOF"])
        GVI_LOSS_FUN = StudentTLoss(dof=gvi_t_dof)
    else:
        GVI_LOSS_FUN = GaussianLoss()

    # Load data
    DATA_PATH = "./data/sim/meas_data_se2.pkl"
    with open(DATA_PATH, "rb") as f:
        data = pickle.load(f)

    x0: State = data["x0"]
    P0: np.ndarray = data["P0"]
    proc_model = data["process_model"]
    fitted_noise_dict = data["fitted_noise_params"]
    landmarks = data["landmarks"]

    gt_data: List[SE2State] = data["ground_truth"]
    gt_data = [s for s in gt_data if s.stamp < T_END]
    gt_stamps = [gt.stamp for gt in gt_data]
    input_data: List[Input] = data["input_data"]
    input_data = [u for u in input_data if u.stamp < T_END]

    meas_data_map: List[Measurement] = data["meas_data_non_gauss"]
    meas_data_map = [m for m in meas_data_map if m.stamp < T_END]

    meas_data_gvi: List[Measurement] = data["meas_data_non_gauss"]
    meas_data_gvi = [m for m in meas_data_gvi if m.stamp < T_END]
    meas_stamps = [meas.stamp for meas in meas_data_map]

    if USE_FIT and isinstance(GVI_LOSS_FUN, SkewLaplaceLoss):
        for i in range(len(meas_data_map)):
            meas_map = meas_data_map[i]
            meas_gvi = meas_data_gvi[i]

            std_dev_gauss = fitted_noise_dict["Gaussian"][1]
            std_dev_gvi = fitted_noise_dict["Skew Laplace"][1]
            skew_lambda_gvi = fitted_noise_dict["Skew Laplace"][2]
            GVI_LOSS_FUN = SkewLaplaceLoss(lamb=skew_lambda_gvi)
            meas_map.model._R = np.array([std_dev_gauss**2])
            meas_gvi.model._R = np.array([std_dev_gvi**2])

    if NOISE:
        x0_state = x0.plus(nav.randvec(P0))

    # MAP Computation
    print("Starting MAP Estimation")
    print("Starting MAP Estimation")
    problem, init_pose_est = construct_planar_map(
        x0=x0_state.copy(),
        P0=P0.copy(),
        input_data=input_data,
        process_model=proc_model,
        meas_data=meas_data_map,
        loss_fun=MAP_LOSS_FUN,
        slam=False,
        step_tol=STEP_TOL,
    )
    # Initialize ESGVI information
    problem.variables = {k: v.copy() for k, v in problem.variables_init.items()}
    problem._compute_size_of_problem()
    _, H, _ = problem.compute_error_jac_cost()
    esgvi_init_info: np.ndarray = (H.T @ H).copy()
    #
    # %%
    if TIME_IT:
        timer = timeit.default_timer
        start_time_map = timer()
        opt_results = problem.solve()
        elapsed_time_map = timer() - start_time_map
        print(f"MAP solved in: {elapsed_time_map:.6f} seconds")
        print(" -------------------------- ")
    else:
        opt_results = problem.solve()

    # print(opt_results["summary"])
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
    gt_stamps = [x.stamp for x in gt_data]
    matches = nav.associate_stamps(estimate_stamps_map, gt_stamps)

    est_list_map = []
    gt_list = []
    for match in matches:
        gt_list.append(gt_data[match[1]])
        est_list_map.append(estimate_list_map[match[0]])

    results_map = nav.GaussianResultList.from_estimates(est_list_map, gt_list)

    ###############################
    # Generate ESGVI Factor Graph
    ###############################
    if MAP_INIT:
        esgvi_graph = esgvi_from_map(
            map_problem=problem,
            proc_cubature="gh",
            meas_cubature="gh",
            cubature_order=3,
            proc_loss=GaussianLoss(),
            meas_loss=GVI_LOSS_FUN,
        )
    else:
        esgvi_graph = generate_esgvi_graph(
            x0_state.copy(),
            P0=P0.copy(),
            init_info_matrix=esgvi_init_info,
            input_data=input_data,
            meas_data=meas_data_gvi,
            process_model=proc_model,
            proc_cubature="gh",
            meas_cubature="gh",
            cubature_order=3,
            meas_loss=GVI_LOSS_FUN,
            proc_loss=GaussianLoss(),
        )
    esgvi_graph.verbose = VERBOSE
    esgvi_graph.max_iters = MAX_ITERS
    esgvi_graph.backtrack_iters = BACK_ITERS
    esgvi_graph.init_step_distance = INIT_STEP_SIZE
    esgvi_graph.step_tol = STEP_TOL

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
    gt_stamps = [float(x.stamp) for x in gt_data]
    matches = nav.associate_stamps(est_stamps_gvi, gt_stamps)
    est_list_gvi = []
    gt_list = []
    for match in matches:
        est_list_gvi.append(est_list_gvi_raw[match[0]])
        gt_list.append(gt_data[match[1]])

    results_gvi = nav.GaussianResultList.from_estimates(est_list_gvi, gt_list)

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
    ax[0].plot(
        results_gvi.stamp, results_gvi.error[:, 0], label="ESGVI", linestyle="--"
    )
    ax[0].fill_between(
        results_gvi.stamp,
        results_gvi.three_sigma[:, 0],
        -results_gvi.three_sigma[:, 0],
        alpha=0.1,
        color="orange",
    )
    ax[1].set_ylabel(r"$x$ (m)")
    ax[1].plot(
        results_gvi.stamp, results_gvi.error[:, 1], label="ESGVI", linestyle="--"
    )
    ax[1].fill_between(
        results_gvi.stamp,
        results_gvi.three_sigma[:, 1],
        -results_gvi.three_sigma[:, 1],
        alpha=0.1,
        color="orange",
    )
    ax[2].set_ylabel(r"$y$ (m)")
    ax[2].plot(
        results_gvi.stamp, results_gvi.error[:, 2], label="ESGVI", linestyle="--"
    )
    ax[2].fill_between(
        results_gvi.stamp,
        results_gvi.three_sigma[:, 2],
        -results_gvi.three_sigma[:, 2],
        alpha=0.1,
        color="orange",
    )
    ax[1].set_xlabel("Time (s)")
    ax[0].legend(loc="upper right")
    ax[1].legend()
    ax[2].legend()
    plt.tight_layout()
    if SAVE_FIGS:
        plt.savefig(
            f"/home/astirl/Documents/courses/assignments/mech_642/gvi_ws/figs/se2_3sigma.pdf"
        )
    if SHOW_FIGS:
        plt.show()

    # Poses Plot
    fig, ax = nav.plot_poses(poses=pose_list_map, step=100, label="MAP")
    fig, ax = nav.plot_poses(pose_list_gvi, step=100, ax=ax, label="ESGVI")
    fig, ax = nav.plot_poses(poses=gt_data, ax=ax, step=None, label="Ground Truth")
    for l in landmarks:
        ax.plot(l[0], l[1], "x")
    ax.set_title("Estimated poses")
    ax.set_xlabel(r"$x$ (m)")
    ax.set_ylabel(r"$y$ (m)")
    ax.legend()
    plt.tight_layout()
    if SAVE_FIGS:
        plt.savefig(
            f"/home/astirl/Documents/courses/assignments/mech_642/gvi_ws/figs/se2_traj.pdf"
        )
    if SHOW_FIGS:
        plt.show()

    # Plot NEES
    fig, axs = nav.plot_nees(results_map, label="MAP", confidence_interval=0.997)
    fig, axs = nav.plot_nees(
        results_gvi, ax=axs, label="ESGVI", confidence_interval=0.997
    )
    axs.set_xlabel("Time (s)")
    axs.set_title("NEES")
    if SAVE_FIGS:
        plt.savefig(
            f"/home/astirl/Documents/courses/assignments/mech_642/gvi_ws/figs/se2_NEES.pdf"
        )
    if SHOW_FIGS:
        plt.show()

    # Comparison table
    print("Estimation Performance")
    print(" Method | RMSE (rad) |  RMSE (m)  |  aNEES")
    print("----------------------------------------")
    print(
        f" ESGVI  |  {np.sqrt(np.mean(np.square(results_gvi.error[:,0]))):.5f}   |  {np.sqrt(np.mean(np.square(results_gvi.error[:,1:]))):.5f}   |  {(np.mean(results_gvi.nees/results_gvi.dof)):.5f}"
    )
    print(
        f" MAP    |  {np.sqrt(np.mean(np.square(results_map.error[:,0]))):.5f}   |  {np.sqrt(np.mean(np.square(results_map.error[:,1:]))):.5f}   |  {(np.mean(results_map.nees/results_map.dof)):.5f}"
    )
    print(" -------------------------- ")
    print(f"Total degrees of freedom x: {esgvi_graph._graph_total_dof}")
    print(f"Poses: {(esgvi_graph._num_poses):.0f}")

# %%

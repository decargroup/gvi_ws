# %%
import os
import sys

# Get the absolute path of the project root (one level above "test")
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Change the working directory to the project root
os.chdir(PROJECT_ROOT)

# Add project root to sys.path so Python finds 'src'
sys.path.insert(0, PROJECT_ROOT)
import numpy as np
import scipy.linalg
import navlie as nav
import timeit
import matplotlib.pyplot as plt

from src.graph.factors import Factor, ProcessFactor, MeasurementFactor, PriorFactor
from src.graph.esgvi import ESGVI
from src.graph.construct_esgvi import generate_trajectory, esgvi_from_map
from src.models.models import LaserRangeFinder
from src.util.psd import (
    force_sym_PSD,
    force_sym,
    regularize,
    fast_positive_definite_inverse,
    isPD,
)
from src.util.sparsity import force_block_banded_sparsity
from src.util.map_batch import construct_planar_map
from navlie.types import State, StateWithCovariance, Measurement, Input
from navlie.lib.states import MatrixLieGroupState, SE2State, VectorState
from navlie.filters import generate_sigmapoints
from navlie.lib.models import (
    BodyFrameVelocity,
    DoubleIntegrator,
    RangePointToAnchor,
    PointRelativePosition,
)
from navlie.batch.residuals import ProcessResidual

from typing import List, Tuple

# %%
if __name__ == "__main__":
    np.random.seed(2)
    T_END = 1.0
    NOISE = True
    CUB_METHOD = "gh"
    CUB_ORDER = 3
    MAP_INIT = False
    TIME_IT = False
    STEP_TOL = 1e-8
    # ESGVI params
    BACKTRACK = False
    VERBOSE = True
    MAX_ITERS = 10
    BACK_ITERS = 1
    INIT_STEP_SIZE = 1e0
    # Script Params
    SAVE_FIGS = True
    SHOW_FIGS = False

    # Init Prior
    x0 = SE2State(value=np.array([0, 0, 0]), stamp=0.0, state_id="x0")
    P0 = np.identity(3) * 1e-2
    # Init landmarks
    landmark_positions = [[2, 1]]
    landmark_states = [
        VectorState(landmark, state_id=f"l{i}")
        for i, landmark in enumerate(landmark_positions)
    ]
    # Init models
    Q_d = np.identity(3) * 0.2
    proc_model = BodyFrameVelocity(Q=Q_d)
    proc_model_freq = 100

    # Meas Model
    R_d = np.identity(2) * 1e-1
    meas_models_gen = [
        PointRelativePosition(
            landmark_position=np.array([l.value]), R=R_d, landmark_id="l0"
        )
        for l in landmark_states
    ]
    # R_d = np.identity(1) * 1e-1
    # meas_models_gen = [
    #     RangePointToAnchor(anchor_position=l.value, R=R_d) for l in landmark_states
    # ]
    meas_model_freq = 10

    # Input Profile
    input_profile = lambda t, x: np.array([np.cos(0.1 * t), 1.0, 0])

    # Data Generation
    dg = nav.DataGenerator(
        proc_model,
        input_profile,
        Q_d,
        input_freq=proc_model_freq,
        meas_model_list=meas_models_gen,
        meas_freq_list=[meas_model_freq] * len(meas_models_gen),
    )
    gt_poses, input_data, meas_data = dg.generate(
        x0.copy(), start=0.0, stop=T_END, noise=NOISE
    )
    # If limit on poses wanted
    input_data_lim = input_data[:]
    meas_data_lim = meas_data[:]
    gt_data_lim = gt_poses[:]

    if NOISE:
        x0_state = x0.plus(nav.randvec(P0))

    # MAP Computation
    print("Starting MAP Estimation")
    problem, init_pose_est = construct_planar_map(
        x0=x0_state.copy(),
        P0=np.copy(P0),
        input_data=input_data_lim,
        process_model=proc_model,
        meas_data=meas_data_lim,
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
    gt_stamps = [x.stamp for x in gt_data_lim]
    matches = nav.associate_stamps(estimate_stamps_map, gt_stamps)

    est_list_map = []
    gt_list = []
    for match in matches:
        gt_list.append(gt_data_lim[match[1]])
        est_list_map.append(estimate_list_map[match[0]])

    results_map = nav.GaussianResultList.from_estimates(est_list_map, gt_list)

    ###############################
    # Generate ESGVI Factor Graph
    ###############################
    if MAP_INIT:
        esgvi_graph = esgvi_from_map(
            map_problem=problem, cubature_method=CUB_METHOD, cubature_order=CUB_ORDER
        )
    else:
        esgvi_graph = generate_trajectory(
            x0_state.copy(),
            P0=P0.copy(),
            init_info_matrix=esgvi_init_info,
            input_data=input_data_lim,
            meas_data=meas_data_lim,
            process_model=proc_model,
            cubature=CUB_METHOD,
            cubature_order=CUB_ORDER,
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
    gt_stamps = [float(x.stamp) for x in gt_data_lim]
    matches = nav.associate_stamps(est_stamps_gvi, gt_stamps)
    est_list_gvi = []
    gt_list = []
    for match in matches:
        est_list_gvi.append(est_list_gvi_raw[match[0]])
        gt_list.append(gt_data_lim[match[1]])

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
    fig, ax = nav.plot_poses(poses=gt_data_lim, ax=ax, step=None, label="Ground Truth")
    for l in landmark_states:
        ax.plot(l.value[0], l.value[1], "x")
    ax.set_title("Estimated poses")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
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
    print("Average Error: ")
    print(" Method | Heading  |    X    |   Y ")
    print("----------------------------------------")
    print(
        f" ESGVI  | {np.mean(results_gvi.error[:,0]):.5f} | {np.mean(results_gvi.error[:,1]):.5f} | {np.mean(results_gvi.error[:,2]):.5f}"
    )
    print(
        f" MAP    | {np.mean(results_map.error[:,0]):.5f} | {np.mean(results_map.error[:,1]):.5f} | {np.mean(results_map.error[:,2]):.5f}"
    )
    print(" -------------------------- ")
    print(f"Total degrees of freedom x: {esgvi_graph._graph_total_dof}")
    print(f"Poses: {(esgvi_graph._num_poses):.0f}")

# %%

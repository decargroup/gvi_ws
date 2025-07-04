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

# %%
if __name__ == "__main__":
    config = load_config("config/se2_localization.yaml")
    noise_config = load_config("config/noise_config.yaml")

    np.random.seed(config["seed"])

    T_END = config["T_END"]
    NOISE = config["NOISE"]
    CUB_METHOD_PROC = config["SP_METHOD_PROC"]
    CUB_METHOD_MEAS = config["SP_METHOD_MEAS"]
    CUB_ORDER = config["CUB_ORDER"]
    MAP_INIT = config["MAP_INIT"]
    TIME_IT = config["TIME_IT"]
    MEAS_MODEL = config["MEAS_MODEL"]

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

    # Init Prior
    x0 = SE2State(value=np.array([0, 0, 0]), stamp=0.0, state_id="x0")
    P0 = np.identity(3) * 1e-3
    # Init landmarks
    landmark_positions = [[2, 1], [0, 1], [2, 0]]
    num_landmarks = len(landmark_positions)
    # landmark_positions = [[2, 1]]
    landmark_states = [
        VectorState(landmark, state_id=f"l{i}")
        for i, landmark in enumerate(landmark_positions)
    ]
    # Init models
    Q_d = np.diag([0.1**2, 0.1, 0.05])
    proc_model = BodyFrameVelocity(Q=Q_d)
    proc_model_freq = 100

    # Meas Model
    meas_model_freq = 10

    if MEAS_MODEL == "relative_pos":
        R_d = np.identity(2) * 1e-2
        meas_models_gen = [
            PointRelativePosition(
                landmark_position=np.array([l.value]), R=R_d, landmark_id=f"l{i}"
            )
            for i, l in enumerate(landmark_states)
        ]
    elif MEAS_MODEL == "range_point":
        R_d = np.identity(1) * 1e-2
        meas_models_gen = [
            RangePointToAnchor(anchor_position=l.value, R=R_d) for l in landmark_states
        ]
    else:  # RangePoseToAnchor
        R_d = np.identity(1) * 1e-2
        tag_position = np.array([0.1, 0.1])
        meas_models_gen = [
            RangePoseToAnchor(
                anchor_position=l.value, tag_body_position=tag_position, R=R_d
            )
            for l in landmark_states
        ]

    # Input Profile
    input_profile = lambda t, x: np.array([np.cos(0.1 * t), 1.0, 0])

    # Data Generation
    dg = DataGenerator(
        process_model=proc_model,
        input_func=input_profile,
        input_covariance=Q_d,
        input_freq=proc_model_freq,
        meas_model_list=meas_models_gen,
        meas_freq_list=[meas_model_freq] * len(meas_models_gen),
        process_noise_type=PROC_NOISE,
        measurement_noise_type=MEAS_NOISE,
    )
    # Gaussian Data Generation
    dg_gaussian = DataGenerator(
        process_model=proc_model,
        input_func=input_profile,
        input_covariance=Q_d,
        input_freq=proc_model_freq,
        meas_model_list=meas_models_gen,
        meas_freq_list=[meas_model_freq] * len(meas_models_gen),
        process_noise_type="gaussian",
        measurement_noise_type="gaussian",
    )

    gt_data, input_data_gauss, meas_data_gauss = dg_gaussian.generate(
        x0.copy(), 0, T_END, noise=NOISE
    )
    gt_poses, input_data, meas_data = dg.generate(
        x0.copy(), start=0.0, stop=T_END, noise=NOISE
    )

    fig, axs = plt.subplots(1, 2, sharey=True)
    fig_gauss, ax_gauss = nav.plot_meas(
        meas_data_gauss[::num_landmarks], state_list=gt_data, axs=axs[0]
    )
    ax_gauss[0].set_title(f"Gaussian Range Measurements")
    ax_gauss[0].set_xlabel(f"Time (s)")
    ax_gauss[0].set_ylabel(f"Range (m)")
    fig_dual, ax_heavy = nav.plot_meas(
        meas_data[::num_landmarks], state_list=gt_data, axs=axs[1]
    )
    ax_heavy[0].set_title(f"{MEAS_NOISE.capitalize()} Range Measurements")
    ax_heavy[0].set_xlabel(f"Time (s)")
    low, up = ax_heavy[0].get_ylim()
    ax_gauss[0].set_ybound(low, up)
    if SAVE_FIGS:
        plt.savefig(
            f"/home/astirl/Documents/courses/assignments/mech_642/gvi_ws/figs/se2_noise_comp.pdf"
        )
    plt.show()

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
            map_problem=problem,
            cubature_method=CUB_METHOD_PROC,
            cubature_order=CUB_ORDER,
        )
    else:
        esgvi_graph = generate_esgvi_graph(
            x0_state.copy(),
            P0=P0.copy(),
            init_info_matrix=esgvi_init_info,
            input_data=input_data_lim,
            meas_data=meas_data_lim,
            process_model=proc_model,
            proc_cubature=CUB_METHOD_PROC,
            meas_cubature=CUB_METHOD_MEAS,
            cubature_order=CUB_ORDER,
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

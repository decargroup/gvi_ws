# %%
import numpy as np
import scipy.linalg
import navlie as nav
import timeit
import matplotlib.pyplot as plt
from typing import List, Tuple

from gvi_ws.graph.factors import Factor, ProcessFactor, MeasurementFactor, PriorFactor
from gvi_ws.graph.esgvi import ESGVI
from gvi_ws.graph.losses import SkewLaplaceLoss, GaussianLoss, StudentTLoss, CauchyLoss
from gvi_ws.graph.construct_esgvi import generate_trajectory, esgvi_from_map
from gvi_ws.util.map_batch import construct_planar_map
from gvi_ws.util.load_config import load_config

from gvi_ws.util.data_generation import DataGenerator
from navlie.lib.states import SE2State, VectorState
from navlie.types import StateWithCovariance
from navlie.lib.models import SingleIntegrator, RangePointToAnchor
from navlie.batch.losses import L2Loss, CauchyLoss

if __name__ == "__main__":
    config = load_config("config/2D_localization.yaml")
    np.random.seed(config["SEED"])
    T_END = config["T_END"]
    NOISE = config["NOISE"]
    CUB_METHOD_PROC = config["SP_METHOD_PROC"]
    CUB_METHOD_MEAS = config["SP_METHOD_MEAS"]
    CUB_ORDER = config["CUB_ORDER"]
    MAP_INIT = config["MAP_INIT"]
    TIME_IT = config["TIME_IT"]
    STEP_TOL = config["STEP_TOL"]
    # ESGVI params
    VERBOSE = config["VERBOSE"]
    MAX_ITERS = config["MAX_ITERS"]
    BACK_ITERS = config["BACK_ITERS"]
    INIT_STEP_SIZE = float(config["INIT_STEP_SIZE"])
    # Noise Params
    PROC_NOISE = config["PROC_NOISE"]
    MEAS_NOISE = config["MEAS_NOISE"]
    MAP_LOSS_FUN = config["MAP_LOSS_FUN"]
    GVI_LOSS_FUN = config["GVI_LOSS_FUN"]
    if MAP_LOSS_FUN == "cauchy":
        MAP_LOSS_FUN = CauchyLoss()
    else:
        MAP_LOSS_FUN = L2Loss()
    if GVI_LOSS_FUN == "skew_laplace":
        gvi_skew_lambda = float(config["GVI_SKEW_LAMBDA"])
        GVI_LOSS_FUN = SkewLaplaceLoss(lamb=gvi_skew_lambda)
    elif GVI_LOSS_FUN == "student_t":
        gvi_t_dof = float(config["GVI_T_DOF"])
        GVI_LOSS_FUN = StudentTLoss(dof=gvi_t_dof)
    else:
        GVI_LOSS_FUN = GaussianLoss()
    
    # Script Params
    SAVE_FIGS = config["SAVE_FIGS"]
    SHOW_FIGS = config["SHOW_FIGS"]

    # Init Prior
    x0 = VectorState(value=np.array([1, 0]), stamp=0.0, state_id="x0")
    P0 = np.identity(2) * 1e-3
    # Init Proc model
    Q_d = np.identity(2) * 0.1
    process_model = SingleIntegrator(Q=Q_d)
    proc_freq = 100
    # Init Meas Model
    R_d = np.identity(1) * 1e-3
    anchor = [0, 4]
    meas_model = RangePointToAnchor(anchor_position=anchor, R=R_d)
    meas_freq = 10
    # Data Generation
    input_profile = lambda t, x: np.array([np.sin(t), np.cos(t)])
    # Gaussian Data Generation
    dg_gaussian = DataGenerator(
        process_model=process_model,
        input_func=input_profile,
        input_covariance=Q_d,
        input_freq=proc_freq,
        meas_model_list=meas_model,
        meas_freq_list=meas_freq,
        process_noise_type="gaussian",
        measurement_noise_type="gaussian",
    )
    # Other Heavy-Tailed Noise Generation
    dg_heavy = DataGenerator(
        process_model=process_model,
        input_func=input_profile,
        input_covariance=Q_d,
        input_freq=proc_freq,
        meas_model_list=meas_model,
        meas_freq_list=meas_freq,
        process_noise_type=PROC_NOISE,
        measurement_noise_type=MEAS_NOISE,
    )

    gt_data, input_data_gauss, meas_data_gauss = dg_gaussian.generate(
        x0.copy(), 0, T_END, noise=NOISE
    )
    _, input_data_heavy, meas_data_heavy = dg_heavy.generate(
        x0.copy(), 0, T_END, noise=NOISE
    )

    fig_gauss, ax_gauss = nav.plot_meas(meas_data_gauss, state_list=gt_data)
    ax_gauss[0].set_title(f"Gaussian Range Measurements")
    ax_gauss[0].set_xlabel(f"Time (s)")
    fig_dual, ax_heavy = nav.plot_meas(meas_data_heavy, state_list=gt_data)
    ax_heavy[0].set_title(f"{MEAS_NOISE.capitalize()} Range Measurements")
    ax_heavy[0].set_xlabel(f"Time (s)")
    low, up = ax_heavy[0].get_ylim()
    ax_gauss[0].set_ybound(low, up)
    plt.show()

    if NOISE:
        x0_state = x0.plus(nav.randvec(P0))

    # MAP Computation
    print("Starting MAP Estimation")
    problem, init_pose_est = construct_planar_map(
        x0=x0_state.copy(),
        P0=np.copy(P0),
        input_data=input_data_heavy,
        process_model=process_model,
        meas_data=meas_data_heavy,
        loss_fun=MAP_LOSS_FUN,
        slam=False,
        step_tol=STEP_TOL,  
    )
    # Initialize ESGVI information
    problem.variables = {k: v.copy() for k, v in problem.variables_init.items()}
    problem._compute_size_of_problem()
    _, H, _ = problem.compute_error_jac_cost()
    esgvi_init_info: np.ndarray = (H.T @ H).copy()
    # Solve MAP Batch
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
    gt_stamps = [x.stamp for x in gt_data]
    matches = nav.associate_stamps(estimate_stamps_map, gt_stamps)

    est_list_map = []
    gt_list = []
    for match in matches:
        gt_list.append(gt_data[match[1]])
        est_list_map.append(estimate_list_map[match[0]])

    results_map = nav.GaussianResultList.from_estimates(est_list_map, gt_list)

    # ESGVI Setup
    if MAP_INIT:
        esgvi_graph = esgvi_from_map(
            map_problem=problem, cubature_method=CUB_METHOD_PROC, cubature_order=CUB_ORDER
        )
    else:
        esgvi_graph = generate_trajectory(
            x0_state.copy(),
            P0=P0.copy(),
            init_info_matrix=esgvi_init_info,
            input_data=input_data_heavy,
            meas_data=meas_data_heavy,
            process_model=process_model,
            proc_cubature=CUB_METHOD_PROC,
            cubature_order=CUB_ORDER,
            meas_loss=GVI_LOSS_FUN,
            proc_loss = GaussianLoss()
        )
    esgvi_graph.verbose = VERBOSE
    esgvi_graph.max_iters = MAX_ITERS
    esgvi_graph.backtrack_iters = BACK_ITERS
    esgvi_graph.init_step_distance = INIT_STEP_SIZE
    esgvi_graph.step_tol = STEP_TOL

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
    ax[0].set_ylabel(r"$x$ (rad)")
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
    ax[1].set_ylabel(r"$y$ (m)")
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
    ax[1].set_xlabel("Time (s)")
    ax[0].legend(loc="upper right")
    ax[1].legend()
    plt.tight_layout()
    plt.show()
    # Plot NEES
    fig, axs = nav.plot_nees(results_map, label="MAP", confidence_interval=0.997)
    fig, axs = nav.plot_nees(
        results_gvi, ax=axs, label="ESGVI", confidence_interval=0.997
    )
    axs.set_xlabel("Time (s)")
    axs.set_title("NEES")
    plt.show()

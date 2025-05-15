# %%
import os
import sys

# Get the absolute path of the project root (one level above "test")
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

# Change the working directory to the project root
os.chdir(PROJECT_ROOT)

# Add project root to sys.path so Python finds 'src'
sys.path.insert(0, PROJECT_ROOT)
import numpy as np
import scipy.linalg
import navlie as nav
import timeit
import matplotlib.pyplot as plt

from navlie import monte_carlo

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


if __name__ == "__main__":
    np.random.seed(1)
    # MC Params
    TRIALS = 10
    # Globals
    T_TRIAL = 2
    CUB_METHOD = "gh"
    CUB_ORDER = 3
    STEP_TOL = 1e-8
    BACK_ITERS = 1
    INIT_STEP_SIZE = 1e0
    SAVE_FIGS = True
    MEAS_MODEL = "rel_pos"

    # ESGVI Params
    MAX_ITERS = 5

    # Trajectory Vals
    X0_TRUE_GVI = SE2State(value=np.array([0, 0, 0]), stamp=0.0, state_id="x0")
    X0_TRUE_MAP = SE2State(value=np.array([0, 0, 0]), stamp=0.0, state_id="x0")
    P0 = np.identity(3) * 1e-3

    # Init landmarks
    landmark_positions = [[2, 1], [0, 1], [2, 0]]
    num_landmarks = len(landmark_positions)
    landmark_states = [
        VectorState(landmark, state_id=f"l{i}")
        for i, landmark in enumerate(landmark_positions)
    ]
    # Init models
    Q_d = np.identity(3) * 0.2
    proc_model = BodyFrameVelocity(Q=Q_d)
    proc_model_freq = 100
    # Meas Model
    if MEAS_MODEL == "range":
        R_d = np.identity(1) * 1e-1

        meas_models_gen = [
            RangePointToAnchor(anchor_position=l.value, R=R_d) for l in landmark_states
        ]
    elif MEAS_MODEL == "rel_pos":
        R_d = np.identity(2) * 1e-1
        meas_models_gen = [
            PointRelativePosition(
                landmark_position=np.array([l.value]), R=R_d, landmark_id="l0"
            )
            for l in landmark_states
        ]
    
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
        meas_freq_list=meas_model_freq,
    )

    def run_esgvi_trial(trial_num: int) -> nav.GaussianResultList:
        np.random.seed(trial_num)
        # print("ESGVI: ", trial_num)
        gt_data, input_data, meas_data = dg.generate(
            X0_TRUE_GVI.copy(), start=0.0, stop=T_TRIAL, noise=True
        )
        x0_check = X0_TRUE_GVI.plus(nav.randvec(P0))
        gvi_problem, init_pose_est = construct_planar_map(
            x0=x0_check.copy(),
            P0=np.copy(P0),
            input_data=input_data,
            process_model=proc_model,
            meas_data=meas_data,
            slam=False,
            step_tol=STEP_TOL,
        )
        # Initialize ESGVI information
        gvi_problem.variables = {k: v.copy() for k, v in gvi_problem.variables_init.items()}
        gvi_problem._compute_size_of_problem()
        _, H, _ = gvi_problem.compute_error_jac_cost()
        esgvi_init_info: np.ndarray = (H.T @ H).copy()
        # Create ESGVI Graph
        esgvi_graph = generate_trajectory(
            x0_check.copy(),
            P0=P0.copy(),
            init_info_matrix=esgvi_init_info,
            input_data=input_data,
            meas_data=meas_data,
            process_model=proc_model,
            proc_cubature=CUB_METHOD,
            cubature_order=CUB_ORDER,
        )
        esgvi_graph.verbose = False
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
        return results_gvi

    def run_map_trial(trial_num: int) -> nav.GaussianResultList:
        np.random.seed(trial_num)
        # print("MAP: ", trial_num)
        gt_data, input_data, meas_data = dg.generate(
            X0_TRUE_MAP.copy(), start=0.0, stop=T_TRIAL, noise=True
        )
        x0_check = X0_TRUE_MAP.plus(nav.randvec(P0))
        problem, init_pose_est = construct_planar_map(
            x0=x0_check.copy(),
            P0=np.copy(P0),
            input_data=input_data,
            process_model=proc_model,
            meas_data=meas_data,
            slam=False,
            step_tol=STEP_TOL,
        )
        problem.verbose = False
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
        return results_map

    results_map = monte_carlo(run_map_trial, num_trials=TRIALS, num_jobs=4)
    results_map_list = results_map.trial_results

    results_gvi = monte_carlo(run_esgvi_trial, num_trials=TRIALS, num_jobs=4)
    results_gvi_list = results_gvi.trial_results
    # %%
    import matplotlib.pyplot as plt

    # Plotting parameters
    plt.rc("text", usetex=True)
    plt.rc("font", family="serif", size=14)
    plt.rc("lines", linewidth=2)
    plt.rc("axes", grid=True)
    plt.rc("grid", linestyle="--")

    fig, ax = nav.plot_nees(
        results=results_map, confidence_interval=0.997, label="MAP",
    )
    fig, ax = nav.plot_nees(
        results=results_gvi, ax=ax, confidence_interval=0.997, label="ESGVI"
    )
    ax.set_ylabel(r"Mahalanobis Distance, $d^2_k$")
    ax.set_xlabel("Time (s)")
    ax.set_title(f"aNEES {TRIALS} trials for {num_landmarks} landmarks.")
    if SAVE_FIGS:
        plt.savefig(
            f"/home/astirl/Documents/courses/assignments/mech_642/gvi_ws/figs/se2_aNEES_{MEAS_MODEL}_{TRIALS}_{T_TRIAL}s.pdf"
        )
    plt.show()

    # fig, ax = nav.plot_error(results=results_map_list[0], label="MAP", color="tab:blue")
    # for i, map_res in enumerate(results_map_list[1:]):
    #     fig, ax = nav.plot_error(
    #         results=map_res, axs=ax, color="tab:blue", bounds=False
    #     )

    # fig, ax = nav.plot_error(
    #     results=results_map_list[0],
    #     axs=ax,
    #     label="ESGVI",
    #     color="tab:orange",
    #     bounds=False,
    # )
    # for i, gvi_res in enumerate(results_gvi_list[1:]):
    #     fig, ax = nav.plot_error(
    #         results=gvi_res, axs=ax, color="tab:orange", bounds=False
    #     )
    # plt.show()

    fig, ax = plt.subplots(3, 1, sharex=True)
    ax: List[plt.Axes] = ax

    ax[0].plot(results_map.rmse[:, 0], label="MAP", color="tab:blue")
    ax[1].plot(results_map.rmse[:, 1], label="MAP", color="tab:blue")
    ax[2].plot(results_map.rmse[:, 2], label="MAP", color="tab:blue")
    ax[0].plot(results_gvi.rmse[:, 0], label="ESGVI", color="tab:orange")
    ax[1].plot(results_gvi.rmse[:, 1], label="ESGVI", color="tab:orange")
    ax[2].plot(results_gvi.rmse[:, 2], label="ESGVI", color="tab:orange")
    ax[0].set_ylabel(r"$\theta$ (rad)")
    ax[1].set_ylabel(r"$x$ (m)")
    ax[2].set_ylabel(r"$y$ (m)")
    ax[2].set_xlabel("Time (s)")
    ax[0].legend()
    ax[1].legend()
    ax[2].legend()
    plt.tight_layout()
    if SAVE_FIGS:
        plt.savefig(
            f"/home/astirl/Documents/courses/assignments/mech_642/gvi_ws/figs/se2_rmse_{TRIALS}_{T_TRIAL}s.pdf"
        )
    plt.show()

    

# %%

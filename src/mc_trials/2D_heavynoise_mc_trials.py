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
from typing import List, Tuple

from src.graph.factors import Factor, ProcessFactor, MeasurementFactor, PriorFactor
from src.graph.esgvi import ESGVI
from src.graph.construct_esgvi import generate_trajectory, esgvi_from_map
from src.util.map_batch import construct_planar_map
from src.util.psd import (
    force_sym_PSD,
    force_sym,
    regularize,
    fast_positive_definite_inverse,
)
from src.models.models import (
    StereoCamera,
)
from src.util.data_generation import DataGenerator
from navlie.lib.states import SE2State, VectorState
from navlie.types import StateWithCovariance
from navlie.lib.models import SingleIntegrator, RangePointToAnchor
from navlie import monte_carlo

if __name__ == "__main__":
    np.random.seed(1)
    # MC Params
    TRIALS = 20
    # Globals
    T_TRIAL = 3.0
    CUB_METHOD = "gh"
    CUB_ORDER = 3
    STEP_TOL = 1e-8
    BACK_ITERS = 1
    INIT_STEP_SIZE = 1e0
    SAVE_FIGS = False
    # NOISE Params
    PROC_NOISE = "gaussian"
    MEAS_NOISE = "cauchy"

    # ESGVI Params
    MAX_ITERS = 5
    # Init Prior
    x0 = VectorState(value=np.array([1, 0]), stamp=0.0, state_id="x0")
    P0 = np.identity(2) * 1e-3
    # Init Proc model
    Q_d = np.identity(2) * 0.1
    process_model = SingleIntegrator(Q=Q_d)
    proc_freq = 100
    # Init Meas Model
    R_d = np.identity(1) * 1e-2
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

    def run_esgvi_trial(trial_num: int) -> nav.GaussianResultList:
        np.random.seed(trial_num)
        gt_data, input_data, meas_data_heavy = dg_heavy.generate(
            x0.copy(), 0, T_TRIAL, noise=True
        )
        x0_state = x0.plus(nav.randvec(P0))
        problem, init_pose_est = construct_planar_map(
            x0=x0_state.copy(),
            P0=np.copy(P0),
            input_data=input_data,
            process_model=process_model,
            meas_data=meas_data_heavy,
            slam=False,
            step_tol=STEP_TOL,
        )
        # Initialize ESGVI information
        problem.variables = {k: v.copy() for k, v in problem.variables_init.items()}
        problem._compute_size_of_problem()
        _, H, _ = problem.compute_error_jac_cost()
        esgvi_init_info: np.ndarray = (H.T @ H).copy()
        esgvi_graph = generate_trajectory(
            x0_state.copy(),
            P0=P0.copy(),
            init_info_matrix=esgvi_init_info,
            input_data=input_data,
            meas_data=meas_data_heavy,
            process_model=process_model,
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
        gt_data, input_data, meas_data_heavy = dg_heavy.generate(
            x0.copy(), 0, T_TRIAL, noise=True
        )
        x0_state = x0.plus(nav.randvec(P0))
        problem, init_pose_est = construct_planar_map(
            x0=x0_state.copy(),
            P0=np.copy(P0),
            input_data=input_data,
            process_model=process_model,
            meas_data=meas_data_heavy,
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

    import matplotlib.pyplot as plt

    # Plotting parameters
    plt.rc("text", usetex=True)
    plt.rc("font", family="serif", size=14)
    plt.rc("lines", linewidth=2)
    plt.rc("axes", grid=True)
    plt.rc("grid", linestyle="--")

    fig_map, ax_map = nav.plot_nees(
        results=results_map, confidence_interval=0.997, label="MAP"
    )
    fig, ax = nav.plot_nees(
        results=results_gvi, ax=ax_map, confidence_interval=0.997, label="ESGVI"
    )
    ax.set_xlabel("Time (s)")
    ax.set_title("NEES")
    if SAVE_FIGS:
        plt.savefig(
            f"/home/astirl/Documents/courses/assignments/mech_642/gvi_ws/figs/2D_{MEAS_NOISE}_nees_{TRIALS}_{T_TRIAL}s.pdf"
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
    ax[0].plot(results_gvi.rmse[:, 0], label="ESGVI", color="tab:orange")
    ax[1].plot(results_gvi.rmse[:, 1], label="ESGVI", color="tab:orange")
    ax[0].set_ylabel(r"$x$ (m)")
    ax[1].set_ylabel(r"$y$ (m)")
    ax[1].set_xlabel("Time (s)")
    ax[0].legend()
    ax[1].legend()
    plt.tight_layout()
    if SAVE_FIGS:
        plt.savefig(
            f"/home/astirl/Documents/courses/assignments/mech_642/gvi_ws/figs/2D_{MEAS_NOISE}_rmse_{TRIALS}_{T_TRIAL}s.pdf"
        )
    plt.show()

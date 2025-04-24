# %%
import os
import sys

# Get the absolute path of the project root (one level above "test")
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
print(PROJECT_ROOT)
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
from src.models.models import (
    LaserRangeFinder,
    Simulator,
    StereoCamera,
    DoubleIntegrator,
)
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
    RangePointToAnchor,
    PointRelativePosition,
)
from navlie.batch.residuals import ProcessResidual

from typing import List, Tuple


if __name__ == "__main__":
    # MC Params
    TRIALS = 10
    SAVE_FIGS = False
    # Globals
    T_TRIAL = 5.0
    CUB_METHOD = "gh"
    CUB_ORDER = 3
    STEP_TOL = 1e-8
    BACK_ITERS = 1
    INIT_STEP_SIZE = 1e0

    # ESGVI Params
    MAX_ITERS = 5

    ######## SIM SETUP ###########
    laser_range_freq = 10
    imu_freq = 100
    sigma_acc_continuous = 0.02
    dt = 1 / imu_freq
    R_k = np.array([0.05])

    R_k = np.array([0.01])
    landmark_pos = np.array([10])
    meas_model = StereoCamera(R_d=R_k, landmark_pos=landmark_pos)

    # Init Value
    X0_VAL = [5, 0]
    P0 = np.eye(2) * 1e-3

    # Simulation
    Simulation = Simulator(t_end=T_TRIAL, freq=imu_freq, x0=X0_VAL)
    # Set Forcing Function
    # Forcing function f(t) = A sin(wt)
    f = lambda t: 1 * np.sin(2 * np.pi * t)
    Simulation.set_forcing_function(f)
    dt = 1 / imu_freq
    Q_d = np.array([[sigma_acc_continuous**2 / dt]])
    proc_model = DoubleIntegrator(Q_d)
    # Generating ground truth
    true_pos, true_vel, true_acc = Simulation.generate_ground_truth()

    # Input Profile
    input_profile = lambda t, x: np.array([np.cos(0.1 * t), 1.0, 0])

    def run_esgvi_trial(trial_num: int) -> nav.GaussianResultList:
        np.random.seed(trial_num)
        meas_pos, _, meas_t = Simulation.generate_measurements(
            sigma_acc=sigma_acc_continuous,
            pos_freq=laser_range_freq,
            acc_freq=imu_freq,
            meas_model=meas_model,
            add_noise=True,
        )
        # Get Navlie formatted data
        gt_data, input_data, meas_data = Simulation.get_nav_info()
        x0_state = VectorState(value=np.array(X0_VAL), stamp=gt_data[0].stamp)

        x0_check = x0_state.plus(nav.randvec(P0))
        problem, init_pose_est = construct_planar_map(
            x0=x0_check.copy(),
            P0=np.copy(P0),
            input_data=input_data,
            process_model=proc_model,
            meas_data=meas_data,
            slam=False,
            step_tol=STEP_TOL,
        )
        # Initialize ESGVI information
        problem.variables = {k: v.copy() for k, v in problem.variables_init.items()}
        problem._compute_size_of_problem()
        _, H, _ = problem.compute_error_jac_cost()
        esgvi_init_info: np.ndarray = (H.T @ H).copy()
        # Create ESGVI Graph
        esgvi_graph = generate_trajectory(
            x0_check.copy(),
            P0=P0.copy(),
            init_info_matrix=esgvi_init_info,
            input_data=input_data,
            meas_data=meas_data,
            process_model=proc_model,
            cubature=CUB_METHOD,
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
        meas_pos, _, meas_t = Simulation.generate_measurements(
            sigma_acc=sigma_acc_continuous,
            pos_freq=laser_range_freq,
            acc_freq=imu_freq,
            meas_model=meas_model,
            add_noise=True,
        )
        # Get Navlie formatted data
        gt_data, input_data, meas_data = Simulation.get_nav_info()
        x0_state = VectorState(value=np.array(X0_VAL), stamp=gt_data[0].stamp)

        x0_check = x0_state.plus(nav.randvec(P0))
        problem, init_pose_est = construct_planar_map(
            x0=x0_check.copy(),
            P0=np.copy(P0),
            input_data=input_data,
            process_model=proc_model,
            meas_data=meas_data,
            slam=False,
            step_tol=STEP_TOL,
        )
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

    results_gvi = monte_carlo(run_esgvi_trial, num_trials=TRIALS, num_jobs=4)

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
    plt.savefig(
        f"/home/astirl/Documents/courses/assignments/mech_642/gvi_ws/figs/2D_aNEES.pdf"
    )
    plt.show()

    fig, ax = plt.subplots(3, 1, sharex=True)
    ax: List[plt.Axes] = ax

    ax[0].plot(results_map.rmse[:, 0], label="MAP", color="tab:blue")
    ax[1].plot(results_map.rmse[:, 1], label="MAP", color="tab:blue")
    ax[0].plot(results_gvi.rmse[:, 0], label="ESGVI", color="tab:orange")
    ax[1].plot(results_gvi.rmse[:, 1], label="ESGVI", color="tab:orange")
    ax[0].set_ylabel(r"$x$ (m)")
    ax[1].set_ylabel(r"$\dot{x}$ (m/s)")
    ax[2].set_xlabel("Time (s)")
    ax[0].legend()
    ax[1].legend()
    ax[2].legend()
    plt.tight_layout()
    if SAVE_FIGS:
        plt.savefig(
            f"/home/astirl/Documents/courses/assignments/mech_642/gvi_ws/figs/2D_rmse_{TRIALS}_{T_TRIAL}s.pdf"
        )
    plt.show()

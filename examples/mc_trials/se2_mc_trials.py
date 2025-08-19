# %%
import numpy as np
import scipy.linalg
import navlie as nav
import timeit
import matplotlib.pyplot as plt
import os
import pickle
from navlie import monte_carlo

from gvi_ws.graph.construct_esgvi import generate_esgvi_graph, esgvi_from_map
from gvi_ws.util.map_batch import construct_planar_map, construct_gmm_map
from navlie.types import State, StateWithCovariance, Measurement, Input
from navlie.lib.states import MatrixLieGroupState, SE2State, VectorState
from navlie.lib.models import (
    BodyFrameVelocity,
    RangePointToAnchor,
    PointRelativePosition,
    RangePoseToAnchor,
)
from navlie.batch.losses import CauchyLoss, L2Loss
from gvi_ws.util.load_config import load_config
from gvi_ws.graph.losses import SkewLaplaceLoss, GaussianLoss, StudentTLoss
from typing import List, Tuple
from gvi_ws.util.data_generation import DataGenerator


if __name__ == "__main__":
    # MC Params
    MC_TRIALS = 50
    T_TRIAL = 3.5
    np.random.seed(0)
    MEAS_MODEL = "range_pose"
    MEAS_NOISE = "skew_laplace"

    # Flags to control whether each result should be updated
    update_gmm = True
    update_map = False
    update_gvi = False

    folder = "./data/results/monte_carlo"
    os.makedirs(folder, exist_ok=True)

    filename = f"mc_{int(MC_TRIALS)}_trials_{T_TRIAL}s.pkl"
    save_path = os.path.join(folder, filename)

    # Load existing results if file exists
    if os.path.exists(save_path):
        with open(save_path, "rb") as f:
            existing_results = pickle.load(f)
    else:
        existing_results = {}

    config = load_config("config/se2_localization.yaml")
    noise_config = load_config("config/noise_config.yaml")

    # Load GMM fit parameters
    DATA_PATH = "./data/sim/meas_data_se2.pkl"
    with open(DATA_PATH, "rb") as f:
        data = pickle.load(f)
    fitted_noise_dict = data["fitted_noise_params"]

    # Simulation Params
    np.random.seed(config["SEED"])
    NOISE = config["NOISE"]
    USE_FIT = config["USE_FIT"]
    TIME_IT = config["TIME_IT"]
    STEP_TOL = float(config["STEP_TOL"])

    # ESGVI Params
    VERBOSE = config["VERBOSE"]
    MAX_ITERS = config["MAX_ITERS"]
    BACK_ITERS = config["BACK_ITERS"]
    INIT_STEP_SIZE = float(config["INIT_STEP_SIZE"])
    CUB_METHOD_PROC = config["SP_METHOD_PROC"]
    CUB_METHOD_MEAS = config["SP_METHOD_MEAS"]
    CUB_ORDER = config["CUB_ORDER"]
    MAP_INIT = config["MAP_INIT"]

    # Loss Params
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

    # Trajectory Vals
    X0_TRUE_GVI = SE2State(value=np.array([0, 0, 0]), stamp=0.0, state_id="x0")
    X0_TRUE_MAP = SE2State(value=np.array([0, 0, 0]), stamp=0.0, state_id="x0")
    X0_TRUE_GMM = SE2State(value=np.array([0, 0, 0]), stamp=0.0, state_id="x0")
    P0 = np.identity(3) * 1e-5

    # Init landmarks
    landmark_positions = [[2, 1], [0, 1], [2, 0]]
    num_landmarks = len(landmark_positions)
    landmark_states = [
        VectorState(landmark, state_id=f"l{i}")
        for i, landmark in enumerate(landmark_positions)
    ]
    # Init models
    Q_d = np.diag([0.1**2, 0.1, 0.05])
    proc_model = BodyFrameVelocity(Q=Q_d)
    proc_model_freq = 100
    # Meas Model
    if MEAS_MODEL == "range":
        R_d = np.identity(1) * 1e-2

        meas_models_gen = [
            RangePointToAnchor(anchor_position=l.value, R=R_d) for l in landmark_states
        ]
    elif MEAS_MODEL == "relative_pos":
        R_d = np.identity(2) * 1e-2
        meas_models_gen = [
            PointRelativePosition(
                landmark_position=np.array([l.value]), R=R_d, landmark_id="l0"
            )
            for l in landmark_states
        ]
    elif MEAS_MODEL == "range_pose":
        tag_pos = np.array([0.1, 0.1])
        R_d = np.identity(1) * 1e-2
        meas_models_gen = [
            RangePoseToAnchor(anchor_position=l.value, tag_body_position=tag_pos, R=R_d)
            for l in landmark_states
        ]

    meas_model_freq = 10

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
        process_noise_type="gaussian",
        measurement_noise_type=MEAS_NOISE,
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
            loss_fun=MAP_LOSS_FUN,
        )
        # Initialize ESGVI information
        gvi_problem.variables = {
            k: v.copy() for k, v in gvi_problem.variables_init.items()
        }
        gvi_problem._compute_size_of_problem()
        _, H, _ = gvi_problem.compute_error_jac_cost()
        esgvi_init_info: np.ndarray = (H.T @ H).copy()
        # Create ESGVI Graph
        esgvi_graph = generate_esgvi_graph(
            x0_check.copy(),
            P0=P0.copy(),
            init_info_matrix=esgvi_init_info,
            input_data=input_data,
            meas_data=meas_data,
            process_model=proc_model,
            proc_cubature=CUB_METHOD_PROC,
            meas_cubature=CUB_METHOD_MEAS,
            cubature_order=CUB_ORDER,
            meas_loss=GVI_LOSS_FUN,
            proc_loss=GaussianLoss(),
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
        std_dev_gauss = fitted_noise_dict["Gaussian"][1]
        for meas in meas_data:
            meas.model._R = np.array([std_dev_gauss**2])

        x0_check = X0_TRUE_MAP.plus(nav.randvec(P0))
        problem, init_pose_est = construct_planar_map(
            x0=x0_check.copy(),
            P0=np.copy(P0),
            input_data=input_data,
            process_model=proc_model,
            meas_data=meas_data,
            loss_fun=MAP_LOSS_FUN,
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

    def run_gmm_trial(trial_num: int) -> nav.GaussianResultList:
        np.random.seed(trial_num)
        # print("MAP: ", trial_num)
        # gmm_means = fitted_noise_dict["GMM"][0]
        # gmm_covs = np.square(fitted_noise_dict["GMM"][1])
        # gmm_weights = fitted_noise_dict["GMM"][2]

        gmm_means = np.array([0.06329292, 0.72206901, 0.33444155])
        gmm_covs = np.square(np.array([0.08593795, 0.32732129, 0.14407507]))
        gmm_weights = np.array([0.61686321, 0.09095258, 0.29218421])
        # gmm_means = np.array([0.32488561, 0.06051794, 0.6747335])
        # gmm_covs = np.square(np.array([0.14457706, 0.08320876, 0.35730603]))
        # gmm_weights = np.array([0.29810484, 0.59954923, 0.10234593])

        gt_data, input_data, meas_data = dg.generate(
            X0_TRUE_GMM.copy(), start=0.0, stop=T_TRIAL, noise=True
        )
        x0_check = X0_TRUE_GMM.plus(nav.randvec(P0))
        problem, init_pose_est = construct_gmm_map(
            x0=x0_check.copy(),
            P0=np.copy(P0),
            input_data=input_data,
            process_model=proc_model,
            meas_data=meas_data,
            means=gmm_means,
            covariances=gmm_covs,
            weights=gmm_weights,
            step_tol=STEP_TOL,
        )
        problem.verbose = False
        opt_results = problem.solve()
        variables_opt = opt_results["variables"]
        estimate_list_gmm: List[nav.types.StateWithCovariance] = []
        pose_list_gmm: List[SE2State] = []
        for pose in init_pose_est:
            state = variables_opt[pose.state_id]
            # Extract the covariance for only this current pose state
            cov = problem.get_covariance_block(pose.state_id, pose.state_id)
            estimate_list_gmm.append(StateWithCovariance(state, cov))
            pose_list_gmm.append(state)

        est_stamps = [float(x.state.stamp) for x in estimate_list_gmm]
        gt_stamps = [x.stamp for x in gt_data]
        matches = nav.associate_stamps(est_stamps, gt_stamps)

        est_list_gmm = []
        gt_list = []
        for match in matches:
            gt_list.append(gt_data[match[1]])
            est_list_gmm.append(estimate_list_gmm[match[0]])

        results_gmm = nav.GaussianResultList.from_estimates(est_list_gmm, gt_list)
        return results_gmm

    # Run and update GMM results if needed
    if update_gmm:
        print("Starting MAP (GMM) Monte Carlo")
        results_gmm = monte_carlo(run_gmm_trial, num_trials=MC_TRIALS, num_jobs=-1)
        existing_results["results_gmm"] = results_gmm

    # Run and update MAP results if needed
    if update_map:
        print("Starting MAP (Cauchy) Monte Carlo")
        results_map = monte_carlo(run_map_trial, num_trials=MC_TRIALS, num_jobs=-1)
        existing_results["results_map"] = results_map

    # Run and update GVI results if needed
    if update_gvi:
        print("Starting ESGVI Monte Carlo")
        results_gvi = monte_carlo(run_esgvi_trial, num_trials=MC_TRIALS, num_jobs=-1)
        existing_results["results_gvi"] = results_gvi

    # Save updated results
    with open(save_path, "wb") as f:
        pickle.dump(existing_results, f)


# %%

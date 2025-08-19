# %%
import numpy as np
import navlie as nav
import timeit
import matplotlib.pyplot as plt

from gvi_ws.graph.losses import SkewLaplaceLoss, GaussianLoss, StudentTLoss, CauchyLoss
from gvi_ws.graph.construct_esgvi import generate_esgvi_graph, esgvi_from_map
from gvi_ws.util.data_generation import DataGenerator
from gvi_ws.util.map_batch import construct_planar_map, construct_gmm_map
from gvi_ws.util.load_config import load_config
from navlie.types import State, StateWithCovariance, Measurement, Input
from navlie.lib.states import SE2State

from navlie.batch.losses import CauchyLoss, L2Loss

from typing import List
import pickle
import os

# %%
if __name__ == "__main__":
    config = load_config("config/se2_localization.yaml")
    noise_config = load_config("config/noise_config.yaml")

    # Simulation Params
    np.random.seed(config["SEED"])
    T_END = config["T_END"]
    NOISE = config["NOISE"]
    USE_FIT = config["USE_FIT"]
    TIME_IT = config["TIME_IT"]
    STEP_TOL = float(config["STEP_TOL"])
    SAVE_FIGS = config["SAVE_FIGS"]
    SHOW_FIGS = config["SHOW_FIGS"]

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

    # Set Losses for MAP
    if MAP_LOSS_FUN == "cauchy":
        MAP_LOSS_FUN = CauchyLoss()
    else:
        MAP_LOSS_FUN = L2Loss()

    # Set Losses for ESGVI
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

    meas_data_gmm: List[Measurement] = data["meas_data_non_gauss"]
    meas_data_gmm = [m for m in meas_data_gmm if m.stamp < T_END]

    meas_data_gvi: List[Measurement] = data["meas_data_non_gauss"]
    meas_data_gvi = [m for m in meas_data_gvi if m.stamp < T_END]
    meas_stamps = [meas.stamp for meas in meas_data_map]

    if USE_FIT and isinstance(GVI_LOSS_FUN, SkewLaplaceLoss):
        std_dev_gauss = fitted_noise_dict["Gaussian"][1]
        std_dev_gvi = fitted_noise_dict["Skew Laplace"][1]
        gvi_skew_lambda = fitted_noise_dict["Skew Laplace"][2]
        for i in range(len(meas_data_map)):
            meas_map = meas_data_map[i]
            meas_gvi = meas_data_gvi[i]

            GVI_LOSS_FUN = SkewLaplaceLoss(lamb=gvi_skew_lambda)
            meas_map.model._R = np.array([std_dev_gauss**2])
            meas_gvi.model._R = np.array([std_dev_gvi**2])

    if NOISE:
        x0_state = x0.plus(nav.randvec(P0))

    # MAP Computation
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
    # MAP Setup/Solving
    if config["USE_MAP"]:
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
        if config["EXPORT"]:
            path = "./data/results/se2"
            os.makedirs(path, exist_ok=True)
            data_export = os.path.join(path, "map_results.pkl")
            with open(data_export, "wb") as f:
                pickle.dump(
                    {
                        "results_map": results_map,
                        "sim_time": T_END,
                        "ground_truth": gt_data,
                        "landmarks": landmarks,
                    },
                    f,
                )

    # DO MAP with GMM now
    if config["USE_GVI"]:
        print("Starting MAP GMM Estimation")
        gmm_means = fitted_noise_dict["GMM"][0]
        gmm_covs = np.square(fitted_noise_dict["GMM"][1])
        gmm_weights = fitted_noise_dict["GMM"][2]
        problem_gmm, init_pose_est_gmm = construct_gmm_map(
            x0=x0_state.copy(),
            P0=P0.copy(),
            input_data=input_data,
            process_model=proc_model,
            meas_data=meas_data_gmm,
            means=gmm_means,
            covariances=gmm_covs,
            weights=gmm_weights,
            step_tol=STEP_TOL,
        )
        opt_results_gmm = problem_gmm.solve()
        variables_opt_gmm = opt_results_gmm["variables"]
        estimate_list_gmm: List[nav.types.StateWithCovariance] = []
        for pose in init_pose_est_gmm:
            state = variables_opt_gmm[pose.state_id]
            # Extract the covariance for only this current pose state
            cov = problem.get_covariance_block(pose.state_id, pose.state_id)
            estimate_list_gmm.append(StateWithCovariance(state, cov))

        estimate_stamps_gmm = [float(x.state.stamp) for x in estimate_list_gmm]
        gt_stamps = [x.stamp for x in gt_data]
        matches = nav.associate_stamps(estimate_stamps_gmm, gt_stamps)

        est_list_gmm = []
        gt_list = []
        for match in matches:
            gt_list.append(gt_data[match[1]])
            est_list_gmm.append(estimate_list_gmm[match[0]])

        results_gmm = nav.GaussianResultList.from_estimates(est_list_gmm, gt_list)

        if config["EXPORT"]:
            path = "./data/results/se2"
            os.makedirs(path, exist_ok=True)
            data_export = os.path.join(path, "gmm_results.pkl")
            with open(data_export, "wb") as f:
                pickle.dump(
                    {
                        "results_gmm": results_gmm,
                        "sim_time": T_END,
                        "ground_truth": gt_data,
                        "landmarks": landmarks,
                    },
                    f,
                )

    ###############################
    # Generate ESGVI Factor Graph
    ###############################
    if MAP_INIT:
        # Warmstart from MAP
        esgvi_graph = esgvi_from_map(
            map_problem=problem,
            proc_cubature="gh",
            meas_cubature="gh",
            cubature_order=3,
            proc_loss=GaussianLoss(),
            meas_loss=GVI_LOSS_FUN,
        )
    else:
        # Initialize Factor Graph from scratch
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
    if config["USE_GVI"]:
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

        if config["EXPORT"]:
            path = "./data/results/se2"
            os.makedirs(path, exist_ok=True)
            data_export = os.path.join(path, "gvi_results.pkl")
            with open(data_export, "wb") as f:
                pickle.dump(
                    {
                        "results_gvi": results_gvi,
                        "skew_lambda": gvi_skew_lambda,
                        "sim_time": T_END,
                        "ground_truth": gt_data,
                        "landmarks": landmarks,
                    },
                    f,
                )
            print("Exported ESGVI results.")

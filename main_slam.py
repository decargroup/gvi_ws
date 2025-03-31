# %%
# General Imports
import numpy as np
import navlie as nav
from typing import List
import matplotlib.pyplot as plt
import timeit

# navlie Imports
from navlie.lib import VectorState, SE2State
from navlie.types import StateWithCovariance
from navlie.lib.models import (
    PointRelativePositionSLAM,
    PointRelativePosition,
    BodyFrameVelocity,
    RangePointToAnchor,
)

# pymlg Imports
from pymlg.numpy.se2 import SE2, SO2

from mlg_gvi import GVI
from mlg_factors import (
    slam_factors_from_map,
    construct_slam_factor_list,
    PriorFactor,
    LandmarkPriorFactor,
)
from util.map_batch import construct_planar_map, extract_landmark_est

if __name__ == "__main__":
    np.random.seed(1)
    T_END = 0.03
    TIME_IT = False
    NOISE = True
    MAP_INIT = False
    BACKTRACK = True
    MAP_LANDMARK_PRIOR = False
    GVI_LANDMARK_PRIOR = False
    INIT_ALPHA = 1
    CUB_METHOD = "GH"  # 'spherical' #
    GH_DEG = 3
    GVI_MAX_ITERS = 10
    BACKTRACK_ITERS = 5
    POSE_KEY_STR = "x"
    LANDMARK_KEY_STR = "l"
    DIR = "right"

    # Landmark Setup Generation
    # landmark_positions = [[2,1], [0,1]]
    landmark_positions = [[2, 1]]
    landmark_states = [
        VectorState(landmark, state_id=f"{LANDMARK_KEY_STR}{i}")
        for i, landmark in enumerate(landmark_positions)
    ]

    # Meas Model
    R_d = np.identity(2) * 1e-1
    meas_models_gen = [
        PointRelativePosition(landmark_position=l.value, R=R_d, landmark_id=l.state_id)
        for l in landmark_states
    ]
    meas_model_freq = 10

    # Process Model
    Q_d = np.identity(3) * 0.2
    process_model = BodyFrameVelocity(Q=Q_d)
    proc_model_freq = 100

    # Input Profile
    input_profile = lambda t, x: np.array([np.cos(0.1 * t), 1.0, 0])

    # Initial Conditions
    x0_state = SE2State(
        value=np.array([0, 0, 0]), stamp=0, state_id=f"{POSE_KEY_STR}{0}", direction=DIR
    )
    x0_state_gvi = SE2State(
        value=np.array([0, 0, 0]), stamp=0, state_id=f"{POSE_KEY_STR}{0}", direction=DIR
    )
    P0 = np.identity(x0_state.dof) * 1e-3
    P0_landmark = np.identity(landmark_states[0].dof) * 1e-1
    x0 = StateWithCovariance(state=x0_state_gvi.copy(), covariance=np.copy(P0))

    # Data Generation
    dg = nav.DataGenerator(
        process_model,
        input_profile,
        Q_d,
        input_freq=proc_model_freq,
        meas_model_list=meas_models_gen,
        meas_freq_list=[meas_model_freq] * len(meas_models_gen),
    )
    gt_poses, input_data, meas_data = dg.generate(
        x0_state.copy(), start=0.0, stop=T_END, noise=NOISE
    )

    if NOISE:
        x0_state = x0_state.plus(nav.randvec(P0))
        x0 = StateWithCovariance(state=x0_state.copy(), covariance=np.copy(P0))
        landmarks_perturb = [
            StateWithCovariance(
                landmark.plus(nav.randvec(P0_landmark)), covariance=P0_landmark
            )
            for landmark in landmark_states
        ]

    input_data_lim = input_data[:]
    meas_data_lim = meas_data[:]
    gt_data_lim = gt_poses[:]
    # gt_data_lim = gt_poses[0:2]
    # input_data_lim = input_data[0:2]
    # meas_data_lim = meas_data[0:1]

    # Dimensions
    state_dof = x0_state.dof
    total_state_dof = len(input_data_lim) * state_dof
    landmark_dof = landmarks_perturb[0].state.dof
    total_landmark_dof = len(landmarks_perturb) * landmark_dof
    total_dof = total_state_dof + total_landmark_dof

    # MAP Computation
    print("Starting MAP Estimation")
    problem, init_pose_est = construct_planar_map(
        x0=x0_state.copy(),
        P0=np.copy(P0),
        input_data=input_data_lim,
        process_model=process_model,
        meas_data=meas_data_lim,
        slam=True,
        init_landmark=landmarks_perturb,
        use_landmark_prior=MAP_LANDMARK_PRIOR,
    )

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
    landmark_est_map = []
    for match in matches:
        gt_list.append(gt_data_lim[match[1]])
        est_list_map.append(estimate_list_map[match[0]])
    landmark_est_map = extract_landmark_est(variables_opt, problem, landmark_states)

    # %%
    # GVI Initialization
    if MAP_INIT:
        factored_state_list = slam_factors_from_map(
            opt_variables=variables_opt,
            problem=problem,
            input_data=input_data_lim,
            meas_data=meas_data_lim,
            landmark_data=landmark_states,
            proc_model=process_model,
            meas_model=PointRelativePositionSLAM,
            use_prior=GVI_LANDMARK_PRIOR,
            cubature_type=CUB_METHOD,
            gh_deg=GH_DEG,
        )
        INIT_ALPHA = 5e-6
        BACKTRACK_ITERS = 10

    else:
        factored_state_list = construct_slam_factor_list(
            x0=x0,
            input_data=input_data_lim,
            meas_data=meas_data_lim,
            landmark_data=landmarks_perturb,
            proc_model=process_model,
            meas_model=PointRelativePositionSLAM,
            use_prior=GVI_LANDMARK_PRIOR,
            cubature_type=CUB_METHOD,
            gh_deg=GH_DEG,
        )

    gvi = GVI(
        factored_states=factored_state_list,
        total_dim=total_dof,
        backtrack_on=BACKTRACK,
        debug=True,
        init_alpha=INIT_ALPHA,
        max_iters=GVI_MAX_ITERS,
        backtrack_iters=BACKTRACK_ITERS,
    )
    if MAP_INIT:
        gvi.from_map(map_covariance=problem.compute_covariance())

    # %%
    # GVI Computation
    print("Starting GVI estimation")
    if TIME_IT:
        elapsed_time = timeit.timeit(gvi.solve, number=1)
        print(f"GVI solved in: {elapsed_time:.6f} seconds")
        print(
            f"Poses: {(total_state_dof/state_dof):.0f} | Landmarks: {(total_landmark_dof/landmark_dof):.0f}"
        )
        print(f"Total state size x: {gvi.mean.shape[0]}")
        print(" -------------------------- ")
    else:
        gvi.solve()
        print(
            f"Poses: {(total_state_dof/state_dof):.0f} | Landmarks: {(total_landmark_dof/landmark_dof):.0f}"
        )
        print(f"Total state size x: {gvi.mean.shape[0]}")
        print(" -------------------------- ")

    #####################
    #### Process GVI ####
    #####################
    # %%
    estimate_list_gvi, landmark_est_gvi = gvi.get_estimate_list(get_landmark=True)
    pose_list_gvi = [x.state for x in estimate_list_gvi]
    estimate_stamps = [float(x.stamp) for x in estimate_list_gvi]
    gt_stamps = [x.stamp for x in gt_data_lim]

    matches = nav.associate_stamps(estimate_stamps, gt_stamps)

    est_list_gvi = []
    gt_list = []
    for match in matches:
        gt_list.append(gt_data_lim[match[1]])
        est_list_gvi.append(estimate_list_gvi[match[0]])

    # Postprocess the results and plot
    results_map = nav.GaussianResultList.from_estimates(est_list_map, gt_list)
    results_gvi = nav.GaussianResultList.from_estimates(est_list_gvi, gt_list)

    #####################
    ##### PLOT GVI ######
    #####################

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
    # plt.savefig(f'/home/astirl/Documents/courses/assignments/mech_642/gvi_ws/figs/slam_se2_3sigma.pdf')

    # Poses Plot
    fig, ax = nav.plot_poses(poses=pose_list_map, step=100, label="MAP")
    fig, ax = nav.plot_poses(pose_list_gvi, step=100, ax=ax, label="ESGVI")
    fig, ax = nav.plot_poses(poses=gt_data_lim, ax=ax, step=None, label="Ground Truth")
    for l in landmark_states:
        ax.plot(l.value[0], l.value[1], "x", color="green")
    fig, ax = nav.utils.plot_landmark_estimates(
        landmark_est_map,
        ax=ax,
        landmark_color="tab:blue",
        edge_color="tab:blue",
        set_bounds=False,
        plot_covariance=True,
    )
    fig, ax = nav.utils.plot_landmark_estimates(
        landmark_est_gvi,
        ax=ax,
        landmark_color="tab:orange",
        edge_color="tab:orange",
        set_bounds=False,
        plot_covariance=True,
    )
    ax.set_title("Estimated poses")
    ax.grid(visible=True)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend()
    plt.tight_layout()
    # plt.savefig(f'/home/astirl/Documents/courses/assignments/mech_642/gvi_ws/figs/slam_se2_traj.pdf')
    plt.show()

    # Plot NEES
    fig, axs = nav.plot_nees(results_map, label="MAP", confidence_interval=0.997)
    fig, axs = nav.plot_nees(
        results_gvi, ax=axs, label="ESGVI", confidence_interval=0.997
    )
    axs.set_xlabel("Time (s)")
    axs.set_title("NEES")
    # plt.savefig(f'/home/astirl/Documents/courses/assignments/mech_642/gvi_ws/figs/slam_se2_NEES.pdf')

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
# %%

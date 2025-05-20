import numpy as np
import scipy.linalg
import navlie as nav
from typing import List

from gvi_ws.util.map_batch import construct_planar_map
from gvi_ws.graph.factors import Factor, ProcessFactor, MeasurementFactor, PriorFactor
from gvi_ws.graph.esgvi import ESGVI
from gvi_ws.graph.construct_esgvi import generate_trajectory
from gvi_ws.models.models import LaserRangeFinder
from gvi_ws.util.psd import (
    forgvi_wssym_PSD,
    force_sym,
    regularize,
    fast_positive_definite_inverse,
)
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


def test_build_graph(end_time, noise=True, verbose=False, method="gh", order=3):
    np.random.seed(1)
    T_END = end_time
    NOISE = True
    CUB_METHOD = "gh"
    CUB_ORDER = 3
    MAP_INIT = False
    TIME_IT = False
    # ESGVI params
    BACKTRACK = False
    VERBOSE = True
    MAX_ITERS = 10
    BACK_ITERS = 50
    INIT_STEP_SIZE = 1e-10
    # Init Prior
    x0 = SE2State(value=np.array([0, 0, 0]), stamp=0.0, state_id="x0")
    P0 = np.identity(3) * 1e-5
    # Init landmarks
    landmark_positions = [[2, 1]]
    landmark_states = [
        VectorState(landmark, state_id=f"l{i}")
        for i, landmark in enumerate(landmark_positions)
    ]
    # Init models
    Q_d = np.identity(3) * 0.3
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
    problem, init_pose_est = construct_planar_map(
        x0=x0_state.copy(),
        P0=np.copy(P0),
        input_data=input_data_lim,
        process_model=proc_model,
        meas_data=meas_data_lim,
        slam=False,
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
        input_data=input_data_lim,
        meas_data=meas_data_lim,
        process_model=proc_model,
        proc_cubature=CUB_METHOD,
        cubature_order=CUB_ORDER,
    )
    print("-----------------\nSizing Test: ")
    assert (
        esgvi_graph._num_landmarks == 0
    ), "There should be zero landmarks in localization example"

    assert esgvi_graph._num_poses == len(
        input_data_lim
    ), "Wrong number of poses/states in graph."

    assert (
        len(esgvi_graph.factor_list) == 1 + len(meas_data_lim) + len(input_data_lim) - 1
    ), "Wrong number of factors in graph."

    print("Sizing Test Passed!")

    print(
        "-----------------\nInformation Test: Inverse of the information matrix should be equal to the dead-reckoned covariance."
    )
    assert np.allclose(
        scipy.linalg.pinv(esgvi_graph._information_matrix),
        esgvi_graph._covariance_matrix,
    )
    print("Information Test Passed!")

    esgvi_graph.verbose = False
    esgvi_graph.backtrack_iters = 1
    esgvi_graph.solve()
    print("-----------------\nBacktrack Test: Step size = 1 creates no cost change.")
    assert np.allclose(esgvi_graph.new_cost, esgvi_graph._backtrack_cost)
    print("Backtrack Test Passed!")

    print(
        "-----------------\nFactor Covariance Test: Factor covariance by slicing should be equal to projected covariance."
    )
    covar_matrix = esgvi_graph._covariance_matrix.copy()
    for factor in esgvi_graph.factor_list:
        if isinstance(factor, ProcessFactor):
            x_km1_covar = covar_matrix[factor.state_slices[0], factor.state_slices[0]]
            x_k_covar = covar_matrix[factor.state_slices[1], factor.state_slices[1]]
            cross_covar = covar_matrix[factor.state_slices[0], factor.state_slices[1]]

            # Form factor level covariance and information
            factor_covar = np.block(
                [[x_km1_covar, cross_covar], [cross_covar.T, x_k_covar]]
            )
        else:
            factor_covar = covar_matrix[factor.state_slices[0], factor.state_slices[0]]

        factor_covar_check = factor.projection @ covar_matrix @ factor.projection.T
        assert np.allclose(
            factor_covar_check, factor_covar
        ), f"Sliced covariance: {factor_covar} \n Not equal to projected covariance: {factor_covar_check}, {factor_covar_check == factor_covar}"
    print("Factor Covariance Test Passed!")


def test_esgvi_update():

    # Init Prior
    x0 = SE2State(value=np.array([0, 0, 0]), stamp=0.0, state_id="x0")
    P0 = np.identity(3) * 1e-5

    esgvi_graph = ESGVI()
    esgvi_graph.add_state(key=x0.state_id, variable=x0.copy())
    esgvi_graph.init_covariance(scipy.linalg.inv(P0))
    prev_states = {k: v.copy() for k, v in esgvi_graph.init_states.items()}
    esgvi_graph.states = {k: v.copy() for k, v in esgvi_graph.init_states.items()}
    states_init = [v.copy() for _, v in esgvi_graph.init_states.items()]
    new_states, _, _ = esgvi_graph.update_states(
        delta_mean=np.array([0, 1, 1]).reshape((-1, 1)),
        old_states=prev_states,
        information=np.identity(3),
        covariance=np.identity(3),
    )
    states_post = [v.copy() for v in new_states.values()]
    print(
        "-----------------\nESGVI Update Test: State values should change after update."
    )
    assert np.allclose(
        states_post[0].position, np.array([1, 1])
    ), f"No change in state after update. Post Update value: \n{states_post[0].value} Initial value: \n {states_init[0].value}"

    print("ESGVI Update Test Passed!")


def test_esgvi_backtrack():
    # Init Prior
    x0 = SE2State(value=np.array([0, 0, 0]), stamp=0.0, state_id="x0")
    P0 = np.identity(3) * 1e-5

    esgvi_graph = ESGVI()
    esgvi_graph.add_state(key=x0.state_id, variable=x0.copy())
    esgvi_graph.init_covariance(scipy.linalg.inv(P0))
    prev_states = {k: v.copy() for k, v in esgvi_graph.init_states.items()}
    esgvi_graph._prev_states = {k: v.copy() for k, v in esgvi_graph.init_states.items()}
    states_init = [v.copy() for _, v in esgvi_graph.init_states.items()]
    cost = esgvi_graph.calculate_cost(
        states_init, information=np.identity(6), covariance=np.identity(6)
    )
    esgvi_graph.prev_cost = cost
    esgvi_graph.new_cost = 100
    new_states, _, _ = esgvi_graph.update_states(
        delta_mean=np.array([0, 1, 1]).reshape((-1, 1)),
        old_states=prev_states,
        information=np.identity(3),
        covariance=np.identity(3),
    )
    esgvi_graph.verbose = False
    esgvi_graph.backtrack_iters = 2
    esgvi_graph._delta_mean = np.array([0, 1, 1]).reshape((-1, 1))
    esgvi_graph.backtrack(
        np.identity(3), np.identity(3), init_step_dist=0.5, alpha_multiplier=0.5
    )

    print(
        "-----------------\nESGVI Backtrack Test: State values should change after backtrack."
    )
    init_pos = esgvi_graph._backtrack_states["x0"].position
    assert np.all(
        init_pos == np.array([0.25, 0.25])
    ), f"For 2 backtracking iterations, position should [0.25, 0.25] instead of {init_pos}"

    print("ESGVI Backtrack Test Passed!")


if __name__ == "__main__":
    # test_build_graph(end_time=0.05)
    # test_esgvi_update()
    test_esgvi_backtrack()

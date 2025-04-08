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
from src.graph.factors import Factor, ProcessFactor, MeasurementFactor, PriorFactor
from src.graph.esgvi import ESGVI
from src.graph.construct_esgvi import generate_trajectory
from src.models.models import LaserRangeFinder
from src.util.psd import (
    force_sym_PSD,
    force_sym,
    regularize,
    fast_positive_definite_inverse,
)
from navlie.types import State, StateWithCovariance, Measurement, Input
from navlie.lib.states import MatrixLieGroupState, SE2State, VectorState
from navlie.filters import generate_sigmapoints
from navlie.lib.models import BodyFrameVelocity, DoubleIntegrator, RangePointToAnchor
from navlie.batch.residuals import ProcessResidual


def test_build_graph(end_time, noise=True, verbose=False, method="gh", order=3):
    np.random.seed(1)
    # Init Prior
    x0 = SE2State(value=np.array([0, 0, 0]), stamp=0.0, state_id="x0")
    P0 = np.identity(3) * 0.01
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

    R_d = np.identity(1) * 1e-1
    meas_models_gen = [
        RangePointToAnchor(anchor_position=l.value, R=R_d) for l in landmark_states
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
        meas_freq_list=[meas_model_freq] * len(meas_models_gen),
    )
    gt_poses, input_data, meas_data = dg.generate(
        x0.copy(), start=0.0, stop=end_time, noise=noise
    )
    input_data_lim = input_data[:]
    meas_data_lim = meas_data[:]
    gt_data_lim = gt_poses[:]
    # gt_data_lim = gt_poses[0:2]
    # input_data_lim = input_data[0:2]
    # meas_data_lim = meas_data[0:1]

    esgvi_graph = generate_trajectory(
        x0.copy(),
        P0=P0.copy(),
        input_data=input_data_lim,
        meas_data=meas_data_lim,
        process_model=proc_model,
        cubature=method,
        cubature_order=order,
    )

    proj = np.block([np.identity(x0.dof), np.zeros((x0.dof, x0.dof))])
    proj_process = np.block(
        [
            [np.identity(x0.dof), np.zeros((x0.dof, x0.dof))],
            [np.zeros((x0.dof, x0.dof)), np.identity(x0.dof)],
        ]
    )
    assert esgvi_graph._num_landmarks == 0
    assert esgvi_graph._num_states == 2
    assert len(esgvi_graph.factor_list) == 3

    assert np.allclose(esgvi_graph.factor_list[0].projection, proj)
    assert np.allclose(esgvi_graph.factor_list[1].projection, proj_process)
    assert np.allclose(esgvi_graph.factor_list[2].projection, proj)

    print(
        "-----------------\nInformation Test: Inverse of the information matrix should be equal to the dead-reckoned covariance."
    )
    assert np.allclose(
        scipy.linalg.pinv(esgvi_graph._information_matrix),
        esgvi_graph._covariance_matrix,
    )
    print("Information Test Passed!")

    slc = esgvi_graph.state_slices[x0.state_id]
    print(
        "-----------------\nCovariance Test: Top left covariance block should be equal to prior covariance."
    )
    assert np.allclose(P0, esgvi_graph._covariance_matrix[slc, slc])
    print("Covariance Test Passed!")

    esgvi_graph.verbose = False
    esgvi_graph.backtrack_iters = 1
    esgvi_graph.solve()
    print("-----------------\nBacktrack Test: Step size = 1 creates no cost change.")
    assert np.allclose(esgvi_graph.new_cost, esgvi_graph._backtrack_cost)
    print("Backtrack Test Passed!")


if __name__ == "__main__":
    NOISE = True
    T_END = 0.03
    CUBATURE = "gh"
    CUB_ORDER = 3
    VERBOSE = True
    test_build_graph(T_END, NOISE, VERBOSE, method=CUBATURE, order=CUB_ORDER)

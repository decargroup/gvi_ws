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
from src.models.models import LaserRangeFinder
from src.util.psd import force_sym_PSD
from navlie.types import State, StateWithCovariance, Measurement, Input
from navlie.lib.states import MatrixLieGroupState, SE2State, VectorState


def test_mlg_prior_factor(verbose=False):
    key1 = "x0"
    projection = np.identity(3)
    state_list = [
        SE2State.random(stamp=0, state_id="x0"),
        SE2State.random(stamp=0.1, state_id="x1"),
    ]
    var_slices = {"x0": slice(0, state_list[0].dof)}
    prior_fac = PriorFactor(
        keys=key1,
        prior_state=state_list[0].copy(),
        prior_covariance=np.identity(3),
        variable_slices=var_slices,
        projection=projection,
        cubature="spherical",
        order=4,
    )
    total_covariance = np.identity(3)
    total_information = np.identity(3)
    if verbose:
        print((prior_fac._unit_sigma_pts))
        print(prior_fac._weights)
        print(prior_fac._gen_sigma_pts(state_list, total_covariance))
    # Test 1: Instance
    assert isinstance(prior_fac, PriorFactor)

    # Test 2: Derivative Evaluation
    cost, col, matrix = prior_fac.evaluate_derivatives(
        states=state_list, covar_matrix=total_covariance, info_matrix=total_information
    )

    assert col.shape[0] == total_covariance.shape[0]
    assert matrix.shape == total_covariance.shape
    if verbose:
        print(col)
        print(matrix)
        print(cost)

    delta_mean = scipy.linalg.solve(force_sym_PSD(matrix), -col)
    if verbose:
        print(delta_mean)

    assert np.allclose(delta_mean, np.zeros_like(delta_mean))


def test_vec_prior_factor(verbose=False):
    key1 = "x0"
    projection = np.identity(2)
    state_list = [VectorState(value=np.ones((2, 1)), stamp=0.0, state_id="x0")]
    var_slices = {"x0": slice(0, state_list[0].dof)}
    prior_fac = PriorFactor(
        keys=key1,
        prior_state=state_list[0].copy(),
        prior_covariance=np.identity(2),
        variable_slices=var_slices,
        projection=projection,
        cubature="spherical",
        order=3,
    )
    total_covariance = np.identity(2)
    total_information = np.identity(2)
    if verbose:
        print((prior_fac._unit_sigma_pts))
        print(prior_fac._weights)
        print(prior_fac._gen_sigma_pts(state_list, total_covariance))
    # Test 1: Instance
    assert isinstance(prior_fac, PriorFactor)

    # Test 2: Derivative Evaluation
    cost, col, matrix = prior_fac.evaluate_derivatives(
        states=state_list, covar_matrix=total_covariance, info_matrix=total_information
    )

    assert col.shape[0] == total_covariance.shape[0]
    assert matrix.shape == total_covariance.shape
    if verbose:
        print(col)
        print(matrix)
        print(cost)

    delta_mean = scipy.linalg.solve(matrix, -col)
    if verbose:
        print(delta_mean)

    assert np.allclose(delta_mean, np.zeros_like(delta_mean))


def test_meas_factor():
    key1 = "x0"
    projection = np.block([[np.eye(2, 2), np.zeros((2, 2))]])
    state_list = [VectorState(value=np.array([[1], [0]]), stamp=0.0, state_id="x0")]
    meas_model = LaserRangeFinder(R_d=np.array([[0.01]]))
    meas_val = meas_model.evaluate(state_list[0].copy())
    measurement = Measurement(
        value=meas_val, stamp=0.0, model=meas_model, state_id="x0"
    )
    var_slices = {"x0": slice(0, state_list[0].dof)}
    meas_fac = MeasurementFactor(
        keys=key1,
        measurement=measurement,
        variable_slices=var_slices,
        projection=projection,
        order=4,
        cubature="gh",
    )
    # Test 1: Instance
    assert isinstance(meas_fac, MeasurementFactor)

    total_covariance = np.identity(4)
    total_information = np.identity(4)

    # Test 2: Derivative Evaluation
    cost, col, matrix = meas_fac.evaluate_derivatives(
        states=state_list, covar_matrix=total_covariance, info_matrix=total_information
    )
    print(col)
    print(matrix)
    print(cost)
    assert col.shape[0] == total_covariance.shape[0]
    assert matrix.shape == total_covariance.shape
    delta_mean = scipy.linalg.solve(matrix[0:2, 0:2], -col[:2])

    # print(delta_mean)

    assert np.allclose(delta_mean[0], np.zeros_like(delta_mean[0]))


if __name__ == "__main__":
    VERBOSE = True
    # test_mlg_prior_factor(verbose=VERBOSE)
    test_vec_prior_factor(verbose=VERBOSE)

    # test_meas_factor()

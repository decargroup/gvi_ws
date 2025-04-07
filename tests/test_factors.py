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
from src.util.psd import (
    force_sym_PSD,
    force_sym,
    regularize,
    fast_positive_definite_inverse,
)
from navlie.types import State, StateWithCovariance, Measurement, Input
from navlie.lib.states import MatrixLieGroupState, SE2State, VectorState
from navlie.filters import generate_sigmapoints
from navlie.lib.models import BodyFrameVelocity, DoubleIntegrator
from navlie.batch.residuals import ProcessResidual


def test_mlg_prior_factor(verbose=False, method="gh", order=3):
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
        cubature=method,
        order=order,
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

    all_close = np.allclose(delta_mean, np.zeros_like(delta_mean))
    assert all_close
    if all_close:
        print(
            "\t Passed: MLG Prior Factor has no update when initialized to ground-truth."
        )
    else:
        print(
            "\t Failed: MLG Prior Factor has update when initialized to ground-truth."
        )


def test_vec_prior_factor(verbose=False, method="gh", order=3):
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
        cubature=method,
        order=order,
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

    all_close = np.allclose(delta_mean, np.zeros_like(delta_mean))
    assert all_close
    if all_close:
        print(
            "\t Passed: Vec Prior Factor has no update when initialized to ground-truth."
        )
    else:
        print(
            "\t Failed: Vec Prior Factor has update when initialized to ground-truth."
        )


def test_meas_factor(verbose=False, method="gh", order=3):
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
        order=order,
        cubature=method,
    )
    # Test 1: Instance
    assert isinstance(meas_fac, MeasurementFactor)

    total_covariance = np.identity(4)
    total_information = np.identity(4)

    # Test 2: Derivative Evaluation
    cost, col, matrix = meas_fac.evaluate_derivatives(
        states=state_list, covar_matrix=total_covariance, info_matrix=total_information
    )
    if verbose:
        print(col)
        print(matrix)
        print(cost)
    assert col.shape[0] == total_covariance.shape[0]
    assert matrix.shape == total_covariance.shape
    delta_mean = scipy.linalg.solve(matrix[0:2, 0:2], -col[:2])

    # print(delta_mean)

    all_close = np.allclose(delta_mean[0], np.zeros_like(delta_mean[0]))
    assert all_close
    if all_close:
        print(
            "\t Passed: Vec Measurement Factor has no update when initialized to ground-truth."
        )
    else:
        print(
            "\t Failed: Vec Measurement Factor has update when initialized to ground-truth."
        )


def test_mlg_proc_factor(verbose=False, method="gh", order=3):
    key1 = "x0"
    key2 = "x1"
    projection = np.identity(6)
    proc_model = BodyFrameVelocity(Q=np.identity(3) * 0.2)
    u = nav.lib.VectorInput(
        value=np.array([0.76379441, 0.52015384, 0.38702206]), stamp=0.0
    )
    x0 = SE2State(value=np.array([0, 0, 0]), stamp=0, state_id=key1)
    x1 = proc_model.evaluate(x0, u, dt=0.01)
    x1.stamp = 0.01
    state_list = [x0, x1]
    var_slices = {"x0": slice(0, x0.dof, None), "x1": slice(x0.dof, 2 * x0.dof, None)}
    proc_fac = ProcessFactor(
        keys=[key1, key2],
        process_model=proc_model,
        input=u,
        variable_slices=var_slices,
        projection=projection,
        cubature=method,
        order=order,
    )
    x_km1_covariance = np.identity(2) * 0.1
    A = proc_model.jacobian(x=x0, u=u, dt=0.01)
    x_k_covariance = A @ x_km1_covariance @ A.T + proc_model.covariance(
        x=x0, u=u, dt=0.01
    )
    # x_km1_covariance @ A.T
    total_covariance = force_sym_PSD(
        np.block(
            (
                [
                    [x_km1_covariance, x_km1_covariance @ A.T],
                    [A @ x_km1_covariance, x_k_covariance],
                ]
            )
        )
    )
    total_information = scipy.linalg.inv(total_covariance)
    # Test 2: Derivative Evaluation
    cost, col, matrix = proc_fac.evaluate_derivatives(
        states=state_list, covar_matrix=total_covariance, info_matrix=total_information
    )
    if verbose:
        print(col)
        print(matrix)
        print(cost)

    col_shape_test = col.shape[0] == total_covariance.shape[0]
    assert col_shape_test
    if not col_shape_test:
        print("Failed: Column output wrong shape")
    matrix_shape_test = matrix.shape == total_covariance.shape
    assert matrix_shape_test
    if not matrix_shape_test:
        print("Failed: Matrix output wrong shape")

    delta_mean = scipy.linalg.solve(matrix, -col)

    print(delta_mean)

    all_close = np.allclose(delta_mean, np.zeros_like(delta_mean))
    if all_close:
        print(
            "\t Passed: MLG Process Factor has no update when initialized to ground-truth."
        )
    else:
        print(
            "\t Failed: MLG Process Factor has update when initialized to ground-truth."
        )
    assert all_close


def test_vec_proc_factor(verbose=False, method="gh", order=3):
    key1 = "x0"
    key2 = "x1"
    dt = 0.1
    projection = np.identity(4)
    proc_model = DoubleIntegrator(Q=np.identity(1) * 0.01)
    u = nav.lib.VectorInput(value=np.array([-4.17562158]), stamp=0.0)
    x0 = VectorState(value=np.array([0, 0]), stamp=0, state_id=key1)
    x1 = proc_model.evaluate(x0, u, dt=dt)
    x1.stamp = x0.stamp + dt
    state_list = [x0, x1]
    var_slices = {"x0": slice(0, x0.dof, None), "x1": slice(x0.dof, 2 * x0.dof, None)}
    proc_fac = ProcessFactor(
        keys=[key1, key2],
        process_model=proc_model,
        input=u,
        variable_slices=var_slices,
        projection=projection,
        cubature=method,
        order=order,
    )
    x_km1_covariance = np.identity(2) * 0.01
    A = proc_model.jacobian(x=x0, u=u, dt=dt)
    x_k_covariance = A @ x_km1_covariance @ A.T + proc_model.covariance(
        x=x0, u=u, dt=dt
    )
    # x_km1_covariance @ A.T
    total_covariance = force_sym_PSD(
        np.block(
            (
                [
                    [x_km1_covariance, x_km1_covariance @ A.T],
                    [A @ x_km1_covariance, x_k_covariance],
                ]
            )
        ).astype(dtype=np.float64)
    )
    print("Total Covariance: \n", total_covariance)
    print("Condition:", np.linalg.cond(total_covariance))
    print(np.linalg.cond(total_covariance) < 1 / np.finfo(total_covariance.dtype).eps)
    print("Inverse: \n", np.linalg.inv(total_covariance))
    print("Pseudo-Inverse: \n", np.linalg.pinv(total_covariance))
    # total_information = fast_positive_definite_inverse(total_covariance)
    # total_information = scipy.linalg.pinv(total_covariance)
    # print("Total Info: \n", total_information)
    # # total_information = scipy.linalg.solve(total_covariance, np.identity(4))
    # # e = proc_fac._eval_factor(state_list)
    # # print(e)
    # # proc_residual = ProcessResidual(keys=[key1, key2], process_model=proc_model, u=u)
    # # e = proc_residual.evaluate(states=state_list)

    # # Test 2: Derivative Evaluation
    # cost, col, matrix = proc_fac.evaluate_derivatives(
    #     states=state_list, covar_matrix=total_covariance, info_matrix=total_information
    # )

    # matrix = force_sym(matrix)
    # # matrix = regularize(matrix)
    # # matrix = force_sym_PSD(matrix)
    # if verbose:
    #     print("Column: \n", col)
    #     print("Matrix: \n", matrix)
    #     print("Cost: ", cost)

    # col_shape_test = col.shape[0] == total_covariance.shape[0]
    # assert col_shape_test
    # if not col_shape_test:
    #     print("Failed: Column output wrong shape")
    # matrix_shape_test = matrix.shape == total_covariance.shape
    # assert matrix_shape_test
    # if not matrix_shape_test:
    #     print("Failed: Matrix output wrong shape")
    # delta_mean = scipy.linalg.solve(matrix, -col)

    # print(delta_mean)

    # all_close = np.allclose(delta_mean, np.zeros_like(delta_mean))
    # if all_close:
    #     print(
    #         "\t Passed: MLG Process Factor has no update when initialized to ground-truth."
    #     )
    # else:
    #     print(
    #         "\t Failed: MLG Process Factor has update when initialized to ground-truth."
    #     )
    # assert all_close


if __name__ == "__main__":
    VERBOSE = False
    METHOD = "gh"
    ORDER = 4
    # print("Testing Prior Factors.")
    # print("Test 1: Vector valued prior factor.")
    # test_vec_prior_factor(verbose=VERBOSE, method=METHOD, order=ORDER)
    # print("Test 2: MLG valued prior factor.")
    # test_mlg_prior_factor(verbose=VERBOSE, method=METHOD, order=ORDER)
    # print(" ----------------------- \n ")
    # print("Testing Measurement Factors.")
    # print("Test 1: Vector valued measurement factor.")
    # test_meas_factor()
    # print(" ----------------------- \n ")
    print("Testing Process Factors.")
    print("Test 1: Vec valued process factor evaluation at ground-truth.")
    test_vec_proc_factor(verbose=True)

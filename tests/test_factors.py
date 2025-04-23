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
from typing import List
from src.graph.factors import Factor, ProcessFactor, MeasurementFactor, PriorFactor
from src.models.models import LaserRangeFinder, DoubleIntegrator
from src.util.psd import (
    force_sym_PSD,
    force_sym,
    regularize,
    fast_positive_definite_inverse,
)
from navlie.types import State, StateWithCovariance, Measurement, Input
from navlie.lib.states import MatrixLieGroupState, SE2State, VectorState
from navlie.filters import generate_sigmapoints
from navlie.lib.models import BodyFrameVelocity, LinearMeasurement
from navlie.batch.residuals import ProcessResidual, PriorResidual
from navlie.batch.problem import Problem
from navlie.filters import generate_sigmapoints


def test_mlg_prior_factor(verbose=False, method="gh", order=3):
    key1 = "x0"
    projection = np.identity(3)
    state_list = [SE2State(value=np.zeros((3, 1)), stamp=0.0, state_id="x0")]
    var_slices = {"x0": slice(0, state_list[0].dof)}
    P0 = np.identity(state_list[0].dof) * 1
    prior_fac = PriorFactor(
        keys=key1,
        prior_state=state_list[0].copy(),
        prior_covariance=P0.copy(),
        variable_slices=var_slices,
        projection=projection,
        cubature=method,
        order=order,
    )
    total_covariance = P0.copy()
    total_information = force_sym(scipy.linalg.inv(total_covariance))

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
    print("----------\nTesting mean mlg update: ")
    assert (
        all_close
    ), f"Delta_mean is {delta_mean} rather than {np.zeros_like(delta_mean)}."
    print("Passed!")
    print("----------\nTesting information mlg update: ")
    assert np.allclose(
        matrix, total_information
    ), f"Should recover ground truth information: \n{total_information}\n Rather than: \n{matrix}"
    print("Passed!")

    # Fix Factor Cost Tests
    # print("----------\nTesting factor cost: ")
    # assert np.allclose(cost, np.ones_like(cost)), f"Cost is not zero: \n{cost}"
    # print("Passed!")


def test_vec_prior_factor(verbose=False, method="gh", order=3):
    key1 = "x0"
    projection = np.identity(2)
    state_list = [VectorState(value=np.zeros((2, 1)), stamp=0.0, state_id="x0")]
    var_slices = {"x0": slice(0, state_list[0].dof)}
    P0 = np.identity(state_list[0].dof) * 1e-2
    prior_fac = PriorFactor(
        keys=key1,
        prior_state=state_list[0].copy(),
        prior_covariance=P0.copy(),
        variable_slices=var_slices,
        projection=projection,
        cubature=method,
        order=order,
    )
    total_covariance = P0.copy()
    total_information = force_sym(scipy.linalg.inv(total_covariance))
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
    print("----------\nTesting mean vector update: ")
    assert (
        all_close
    ), f"Delta_mean is {delta_mean} rather than {np.zeros_like(delta_mean)}."
    print("Passed!")
    print("----------\nTesting information vector update: ")
    assert np.allclose(
        matrix, total_information
    ), f"Should recover ground truth information: \n{total_information}\n Rather than: \n{matrix}"
    print("Passed!")

    print("----------\nTesting factor cost: ")
    assert np.allclose(cost, np.ones_like(cost)), f"Cost is not zero: \n{cost}"
    print("Passed!")


def test_meas_factor(verbose=False, method="gh", order=3):
    key1 = "x0"
    projection = np.identity(2)
    state_list = [VectorState(value=np.ones((2, 1)), stamp=0.0, state_id="x0")]
    var_slices = {"x0": slice(0, state_list[0].dof)}
    R_k = np.identity(2) * 1e-2
    R_k_inv = force_sym(scipy.linalg.inv(R_k))
    meas_model = LinearMeasurement(C=np.identity(2), R=R_k.copy())
    meas_val = meas_model.evaluate(state_list[0])
    meas = Measurement(value=meas_val, stamp=state_list[0].stamp, model=meas_model)
    P0 = np.identity(state_list[0].dof) * 1e-3
    meas_fac = MeasurementFactor(
        keys=key1,
        measurement=meas,
        variable_slices=var_slices,
        projection=projection,
        cubature=method,
        order=order,
    )
    total_covariance = P0.copy()
    total_information = force_sym(scipy.linalg.inv(total_covariance))

    # Test 1: Instance
    assert isinstance(meas_fac, MeasurementFactor)

    # Test 2: Derivative Evaluation
    cost, col, matrix = meas_fac.evaluate_derivatives(
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
    print("----------\nTesting mean vector update: ")
    assert (
        all_close
    ), f"Delta_mean is {delta_mean} rather than {np.zeros_like(delta_mean)}."
    print("Passed!")
    print("----------\nTesting information vector update: ")
    assert np.allclose(
        matrix, R_k_inv
    ), f"Info update:\n{matrix} \n Not equal to measurement info:\n {R_k_inv}"
    print("Passed!")

    # Fix Factor Cost Tests
    # print("----------\nTesting factor cost: ")
    # assert np.allclose(cost, np.ones_like(cost)), f"Cost should be 1: \n{cost}"
    # print("Passed!")


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
    total_information = force_sym(scipy.linalg.inv(total_covariance))
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


def test_vec_prior_proc_factor(verbose=False, method="gh", order=3):
    key1 = "x0"
    key2 = "x1"
    dt = 0.01
    projection = np.identity(4)
    proj_prior = np.block([[np.identity(2), np.zeros((2, 2))]])
    Q_d = np.identity(1) * 0.04
    proc_model = DoubleIntegrator(Q=Q_d)
    u = nav.lib.VectorInput(value=np.array([-4.01125337]), stamp=0.0)
    x0 = VectorState(value=np.array([0, 0]), stamp=0, state_id=key1)
    P0 = np.identity(2) * 1e-1
    x1 = proc_model.evaluate(x0, u, dt=dt)
    x1.state_id = "x1"
    x1.stamp = x0.stamp + dt
    state_list = [x0, x1]
    var_slices = {"x0": slice(0, x0.dof, None), "x1": slice(x0.dof, 2 * x0.dof, None)}
    var_slices_prior = {"x0": slice(0, x0.dof, None)}
    prior_fac = PriorFactor(
        keys=key1,
        prior_state=x0.copy(),
        prior_covariance=P0.copy(),
        variable_slices=var_slices_prior,
        projection=proj_prior,
        cubature=method,
        order=order,
    )
    if method == "unscented":
        order = 1e-5
        print(f"Unscented Process Factor Kappa = {order}")
    proc_fac = ProcessFactor(
        keys=[key1, key2],
        process_model=proc_model,
        input=u,
        variable_slices=var_slices,
        projection=projection,
        cubature=method,
        order=order,
    )

    fac_list: List[Factor] = [proc_fac, prior_fac]

    problem = Problem()
    problem.add_variable(key1, variable=x0.copy())
    problem.add_variable(key2, variable=x1.copy())
    prior_res = PriorResidual(
        keys=key1, prior_state=x0.copy(), prior_covariance=P0.copy()
    )
    problem.add_residual(prior_res)
    proc_res = ProcessResidual(keys=[key1, key2], process_model=proc_model, u=u)
    problem.add_residual(proc_res)

    # Initialize ESGVI information
    problem.variables = {k: v.copy() for k, v in problem.variables_init.items()}
    problem._compute_size_of_problem()
    _, H, _ = problem.compute_error_jac_cost()
    esgvi_init_info = (H.T @ H).copy()
    total_info = esgvi_init_info.copy()
    total_covar = force_sym(scipy.linalg.inv(total_info))

    cost_prior, col_prior, mat_prior = prior_fac.evaluate_derivatives(
        states=[x0.copy()],
        covar_matrix=total_covar.copy(),
        info_matrix=total_info.copy(),
    )
    print("Testing Two State Vector Prior Factor Info Update: ")
    assert np.allclose(
        mat_prior[0:2, 0:2], force_sym(scipy.linalg.inv(P0)), atol=1e-7
    ), f"Prior Matrix Info Update:\n{mat_prior[0:2,0:2]}\nNot equal to Ground Truth Prior Info:\n{force_sym(scipy.linalg.inv(P0))}."
    print("Passed!")

    cost_proc, col_proc, mat_proc = proc_fac.evaluate_derivatives(
        states=[x0.copy(), x1.copy()],
        covar_matrix=total_covar.copy(),
        info_matrix=total_info.copy(),
    )
    gt_mat_proc = total_info - np.block(
        [
            [force_sym(scipy.linalg.inv(P0)), np.zeros((2, 2))],
            [np.zeros((2, 2)), np.zeros((2, 2))],
        ]
    )
    print("Testing Two State Vector Process Factor Info Update: ")
    assert np.allclose(
        mat_proc, gt_mat_proc, atol=1e-7
    ), f"Proc Matrix Info Update:\n{mat_proc}\nNot equal to Ground Truth Proc Info:\n{gt_mat_proc}."
    total_cost = cost_prior + cost_proc
    total_col = col_prior + col_proc
    total_mat = mat_prior + mat_proc

    delta_mean = scipy.linalg.solve(total_mat, -total_col)
    all_close = np.allclose(delta_mean, np.zeros_like(delta_mean))
    print("----------\nTesting Two-State Mean Vector Update: ")
    assert (
        all_close
    ), f"Delta_mean is {delta_mean} rather than {np.zeros_like(delta_mean)}."
    print("Passed!")

    print("----------\nTesting Two-State Vector Information Update: ")
    assert np.allclose(
        total_mat, total_info
    ), f"Info update:\n{total_mat} \n Not equal to ground-truth info:\n {total_info}"
    print("Passed!")
    # print(f"Info update:\n{total_mat} \nEqual to ground-truth info:\n{total_info}")


def test_mlg_prior_proc_factor(verbose=False, method="gh", order=3):
    key1 = "x0"
    key2 = "x1"
    dt = 0.01
    projection = np.identity(6)
    proj_prior = np.block([[np.identity(3), np.zeros((3, 3))]])

    # Init models
    Q_d = np.identity(3) * 0.2
    proc_model = BodyFrameVelocity(Q=Q_d)
    proc_model_freq = 100
    u = nav.lib.VectorInput(
        stamp=0.0, value=np.array([0.04466406, 1.73355141, -0.80204878])
    )
    # Init Prior
    x0 = SE2State(value=np.array([0, 0, 0]), stamp=0.0, state_id="x0")
    P0 = np.identity(3) * 1e-3

    x1 = proc_model.evaluate(x0, u, dt=dt)
    x1.state_id = "x1"
    x1.stamp = x0.stamp + dt
    state_list = [x0, x1]
    var_slices = {"x0": slice(0, x0.dof, None), "x1": slice(x0.dof, 2 * x0.dof, None)}
    var_slices_prior = {"x0": slice(0, x0.dof, None)}
    prior_fac = PriorFactor(
        keys=key1,
        prior_state=x0.copy(),
        prior_covariance=P0.copy(),
        variable_slices=var_slices_prior,
        projection=proj_prior,
        cubature=method,
        order=order,
    )
    proc_fac = ProcessFactor(
        keys=[key1, key2],
        process_model=proc_model,
        input=u,
        variable_slices=var_slices,
        projection=projection,
        cubature=method,
        order=order,
    )

    fac_list: List[Factor] = [proc_fac, prior_fac]

    problem = Problem()
    problem.add_variable(key1, variable=x0.copy())
    problem.add_variable(key2, variable=x1.copy())
    prior_res = PriorResidual(
        keys=key1, prior_state=x0.copy(), prior_covariance=P0.copy()
    )
    problem.add_residual(prior_res)
    proc_res = ProcessResidual(keys=[key1, key2], process_model=proc_model, u=u)
    problem.add_residual(proc_res)

    # Initialize ESGVI information
    problem.variables = {k: v.copy() for k, v in problem.variables_init.items()}
    problem._compute_size_of_problem()
    _, H, _ = problem.compute_error_jac_cost()
    esgvi_init_info = (H.T @ H).copy()
    total_info = esgvi_init_info.copy()
    total_covar = force_sym(scipy.linalg.inv(total_info))

    cost_prior, col_prior, mat_prior = prior_fac.evaluate_derivatives(
        states=[x0.copy()],
        covar_matrix=total_covar.copy(),
        info_matrix=total_info.copy(),
    )
    print("Testing Two State MLG Prior Factor Info Update: ")
    assert np.allclose(
        mat_prior[0:3, 0:3], force_sym(scipy.linalg.inv(P0)), atol=1e-7
    ), f"Prior Matrix Info Update:\n{mat_prior[0:3,0:3]}\nNot equal to Ground Truth Prior Info:\n{force_sym(scipy.linalg.inv(P0))}."
    print("Passed!")

    cost_proc, col_proc, mat_proc = proc_fac.evaluate_derivatives(
        states=[x0.copy(), x1.copy()],
        covar_matrix=total_covar.copy(),
        info_matrix=total_info.copy(),
    )
    gt_mat_proc = total_info - np.block(
        [
            [force_sym(scipy.linalg.inv(P0)), np.zeros((3, 3))],
            [np.zeros((3, 3)), np.zeros((3, 3))],
        ]
    )
    # print("Testing Two State MLG Process Factor Info Update: ")
    # assert np.allclose(
    #     mat_proc, gt_mat_proc, atol=1e-7
    # ), f"Proc Matrix Info Update:\n{mat_proc}\nNot equal to Ground Truth Proc Info:\n{gt_mat_proc}."
    total_cost = cost_prior + cost_proc
    total_col = col_prior + col_proc
    total_mat = mat_prior + mat_proc

    delta_mean = scipy.linalg.solve(total_mat, -total_col)
    all_close = np.allclose(delta_mean, np.zeros_like(delta_mean))
    print("----------\nTesting Two-State Mean Vector Update: ")
    assert (
        all_close
    ), f"Delta_mean is {delta_mean} rather than {np.zeros_like(delta_mean)}."
    print("Passed!")

    print("----------\nTesting Two-State Vector Information Update: ")
    assert np.allclose(
        total_mat, total_info, atol=1e0
    ), f"Info update:\n{total_mat} \n Not equal to ground-truth info:\n {total_info}"
    print("Passed!")
    print(f"Info update:\n{total_mat} \nEqual to ground-truth info:\n{total_info}")


if __name__ == "__main__":
    np.random.seed(1)
    VERBOSE = False
    METHOD = "gh"
    ORDER = 4
    print("Testing Prior Factors.")
    test_vec_prior_factor(verbose=VERBOSE, method=METHOD, order=ORDER)
    test_mlg_prior_factor(verbose=VERBOSE, method=METHOD, order=ORDER)
    print(" ----------------------- \n ")
    print("Testing Measurement Factors.")
    test_meas_factor(verbose=VERBOSE, method=METHOD, order=ORDER)
    print(" ----------------------- \n ")
    print("Testing Process Factors.")
    test_vec_prior_proc_factor(verbose=True, method=METHOD, order=ORDER)
    # test_mlg_prior_proc_factor(verbose=True, method=METHOD, order=ORDER)

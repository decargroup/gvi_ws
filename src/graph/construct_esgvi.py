import numpy as np
import scipy.linalg
import navlie as nav

from src.graph.esgvi import ESGVI
from src.graph.factors import Factor, PriorFactor, ProcessFactor, MeasurementFactor

from navlie.types import State, Measurement, MeasurementModel, ProcessModel, Input
from navlie.lib import MatrixLieGroupState, VectorState
from navlie.utils import find_nearest_stamp_idx

from navlie.batch.problem import Problem
from navlie.batch.residuals import (
    Residual,
    PriorResidual,
    ProcessResidual,
    MeasurementResidual,
)

from typing import List, Tuple


def generate_trajectory(
    x0: State,
    P0: np.ndarray,
    init_info_matrix: np.ndarray,
    input_data: List[Input],
    meas_data: List[Measurement],
    process_model: ProcessModel,
    cubature: str = "gh",
    cubature_order: int = 3,
    init_landmark: List[State] = None,
    P0_lanmdark: np.ndarray = None,
) -> ESGVI:
    if init_landmark is not None:
        raise NotImplementedError("Figure out how to initialize the covariance matrix")

    esgvi_graph = ESGVI()

    # d.o.f calculations
    total_state_dof = len(input_data) * x0.dof
    total_landmark_dof = 0
    if init_landmark is not None:
        total_landmark_dof += len(init_landmark) * init_landmark[0].dof
    total_esgvi_dof = total_state_dof + total_landmark_dof

    # Dead-Reckoned Initialization
    pose_key_str = "x"
    x0_prior = x0.copy()
    x0_prior.state_id = pose_key_str + "0"
    init_pose_est = [x0_prior]
    x = x0_prior.copy()
    for k in range(len(input_data) - 1):
        u = input_data[k]
        dt = input_data[k + 1].stamp - u.stamp
        x = process_model.evaluate(x, u, dt)
        x.stamp = x.stamp + dt
        x.state_id = pose_key_str + str(k + 1)
        init_pose_est.append(x.copy())

    # Add States to graph
    for state in init_pose_est:
        esgvi_graph.add_state(state.state_id, state)

    est_stamps = [state.stamp for state in init_pose_est]
    # Add landmark states to graph
    # for landmark in init_landmark:
    #     esgvi_graph.add_state(landmark.state_id, landmark.copy())
    # Add Residuals
    state_slices = esgvi_graph.state_slices
    state_proj_empty = np.zeros((x0.dof, total_esgvi_dof))

    # Add prior residual
    prior_proj = state_proj_empty.copy()
    prior_proj[:, 0 : x0.dof] = np.identity(x0.dof)
    prior_factor = PriorFactor(
        keys=x0_prior.state_id,
        prior_state=x0_prior.copy(),
        prior_covariance=P0.copy(),
        variable_slices=state_slices,
        projection=prior_proj,
        cubature=cubature,
        order=cubature_order,
    )
    esgvi_graph.add_factor(prior_factor)
    # Add process residuals
    idx_proc = 0
    for k in range(len(input_data) - 1):
        u = input_data[k]
        key_1 = f"{pose_key_str}{k}"
        key_2 = f"{pose_key_str}{k+1}"
        process_proj = np.zeros((2 * x0.dof, total_esgvi_dof))
        process_proj[:, idx_proc : idx_proc + (2 * x0.dof)] = np.identity(2 * x0.dof)
        process_factor = ProcessFactor(
            keys=[key_1, key_2],
            process_model=process_model,
            input=u,
            variable_slices=state_slices,
            projection=process_proj,
            cubature=cubature,
            order=cubature_order,
        )
        esgvi_graph.add_factor(process_factor)
        idx_proc += x0.dof

    for k, meas in enumerate(meas_data):
        pose_idx = find_nearest_stamp_idx(est_stamps, meas.stamp)
        pose = init_pose_est[pose_idx]
        key_1 = pose.state_id
        meas_proj = state_proj_empty.copy()
        idx_meas = x0.dof * pose_idx
        meas_proj[:, idx_meas : idx_meas + x0.dof] = np.identity(x0.dof)
        meas_factor = MeasurementFactor(
            keys=[key_1],
            measurement=meas,
            variable_slices=state_slices,
            projection=meas_proj,
            cubature=cubature,
            order=cubature_order,
        )
        esgvi_graph.add_factor(meas_factor)

    esgvi_graph.init_covariance(information=init_info_matrix)

    return esgvi_graph


def esgvi_from_map(
    map_problem: Problem, cubature_method: str = "gh", cubature_order: int = 3
):
    esgvi_graph = ESGVI()
    total_dof = map_problem._size_state
    state_dof = 0
    for key, state in map_problem.variables.items():
        esgvi_graph.add_state(key, state.copy())
        if key[0] == "x":
            state_dof = state.dof
    proj_empty = np.zeros((state_dof, total_dof))
    proj_proc_empty = np.zeros((2 * state_dof, total_dof))
    idx_proc = 0
    idx_meas = 0
    for k, res in enumerate(map_problem.residual_list):
        if isinstance(res, PriorResidual):
            prior_cov = res._cov.copy()
            prior_state = res._x0.copy()
            keys_k = res.keys
            var_slices = {}
            proj_prior = proj_empty.copy()
            proj_prior[:, 0 : prior_state.dof] = np.identity(prior_state.dof)
            for key in keys_k:
                var_slices[key] = map_problem.variable_slices[key]
            fac_k = PriorFactor(
                keys=keys_k,
                prior_state=prior_state,
                prior_covariance=prior_cov,
                variable_slices=var_slices,
                projection=proj_prior,
            )
            esgvi_graph.add_factor(fac_k)
        elif isinstance(res, ProcessResidual):
            u = res._u
            proc_model = res._process_model
            keys_k = res.keys
            proj_proc = proj_proc_empty.copy()
            proj_proc[:, idx_proc : idx_proc + 2 * state_dof] = np.identity(
                2 * state_dof
            )
            var_slices = {}
            for key in keys_k:
                var_slices[key] = map_problem.variable_slices[key]
            fac_k = ProcessFactor(
                keys=keys_k,
                process_model=proc_model,
                input=u,
                variable_slices=var_slices,
                projection=proj_proc,
                cubature=cubature_method,
                order=cubature_order,
            )
            esgvi_graph.add_factor(fac_k)
            idx_proc += state_dof
        elif isinstance(res, MeasurementResidual):
            meas = res._y
            keys_k = res.keys
            proj_meas = proj_empty.copy()
            proj_meas[:, idx_meas : idx_meas + state_dof] = np.identity(state_dof)
            var_slices = {}
            for key in keys_k:
                var_slices[key] = map_problem.variable_slices[key]
            fac_k = MeasurementFactor(
                keys=keys_k,
                measurement=meas,
                variable_slices=var_slices,
                projection=proj_meas,
                cubature=cubature_method,
                order=cubature_order,
            )
            esgvi_graph.add_factor(fac_k)
            idx_meas += state_dof
        else:
            raise NotImplementedError(
                f"Haven't implemented factors for residuals of type {type(res)}."
            )

    esgvi_graph.init_covariance(map_problem._information_matrix.toarray())
    return esgvi_graph

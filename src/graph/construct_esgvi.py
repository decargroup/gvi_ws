import numpy as np
import scipy.linalg
import navlie as nav

from src.graph.esgvi import ESGVI
from src.graph.factors import Factor, PriorFactor, ProcessFactor, MeasurementFactor

from navlie.types import State, Measurement, MeasurementModel, ProcessModel, Input
from navlie.lib import MatrixLieGroupState, VectorState
from navlie.utils import find_nearest_stamp_idx

from typing import List, Tuple


def generate_trajectory(
    x0: State,
    P0: np.ndarray,
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
    covariance_matrix = np.zeros((total_esgvi_dof, total_esgvi_dof))

    # Dead-Reckoned Initialization
    pose_key_str = "x"
    x0_prior = x0.copy()
    x0_prior.state_id = pose_key_str + "0"
    init_pose_est = [x0_prior]
    covariance_matrix[0 : x0.dof, 0 : x0.dof] = P0.copy()
    x = x0_prior.copy()
    P = P0.copy()
    idx = x0_prior.dof
    for k in range(len(input_data) - 1):
        u = input_data[k]
        dt = input_data[k + 1].stamp - u.stamp
        A = process_model.jacobian(x, u, dt)
        Q = process_model.covariance(x, u, dt)
        P_k = A @ P @ A.T + Q
        x = process_model.evaluate(x, u, dt)
        x.stamp = x.stamp + dt
        x.state_id = pose_key_str + str(k + 1)
        covariance_matrix[idx - x.dof : idx, idx : idx + x.dof] = P @ A.T
        covariance_matrix[idx : idx + x.dof, idx - x.dof : idx] = A @ P
        covariance_matrix[idx : idx + x.dof, idx : idx + x.dof] = P_k
        idx += x.dof
        P = P_k.copy()
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
    idx = 0
    for k in range(len(input_data) - 1):
        u = input_data[k]
        key_1 = f"{pose_key_str}{k}"
        key_2 = f"{pose_key_str}{k+1}"
        process_proj = np.zeros((2 * x0.dof, total_esgvi_dof))
        process_proj[:, idx : idx + (2 * x0.dof)] = np.identity(2 * x0.dof)
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
        idx += x0.dof

    for k, meas in enumerate(meas_data):
        pose_idx = find_nearest_stamp_idx(est_stamps, meas.stamp)
        pose = init_pose_est[pose_idx]
        key_1 = pose.state_id
        meas_proj = state_proj_empty.copy()
        idx = x0.dof * pose_idx
        meas_proj[:, idx : idx + x0.dof] = np.identity(x0.dof)
        meas_factor = MeasurementFactor(
            keys=[key_1],
            measurement=meas,
            variable_slices=state_slices,
            projection=meas_proj,
            cubature=cubature,
            order=cubature_order,
        )
        esgvi_graph.add_factor(meas_factor)

    esgvi_graph.init_covariance(covariance=covariance_matrix)

    return esgvi_graph

# General Imports
import numpy as np
import navlie as nav
import typing
from typing import List
import matplotlib.pyplot as plt
import timeit

# Navlie Imports
from navlie.lib import VectorState, SE2State, VectorInput
from navlie.types import  StateWithCovariance, ProcessModel, Measurement
from navlie.lib.models import PointRelativePositionSLAM, PointRelativePosition, BodyFrameVelocity
from navlie.batch.problem import Problem
from navlie.batch.residuals import ProcessResidual, PriorResidual, Residual, MeasurementResidual
from navlie.utils import find_nearest_stamp_idx
from pymlg.numpy.se2 import SE2, SO2


# Define the measurement residual, which links a robot state to a landmark
class PointRelativePositionResidual(Residual):
    def __init__(
        self,
        keys: typing.List[typing.Hashable],
        meas: nav.types.Measurement,
    ):
        super().__init__(keys)
        # Store the measurement, where the measurement contains the model
        self.meas = meas
        # Evaluate the square root information a single time since it does not
        # depend on the state in this case
        self.sqrt_information = self.meas.model.sqrt_information([])

    def evaluate(
        self,
        states: typing.List[nav.types.State],
        compute_jacobians: typing.List[bool] = None,
    ) -> typing.Tuple[np.ndarray, typing.List[np.ndarray]]:
        # In this case, states is a list of length two, where the first element
        # should be the robot state and the second element should be the
        # landmark state.

        # To evaluate the measurement model that we previously defined,
        # we need to create a composite state from the list of states
        eval_state = nav.CompositeState(states)

        # Evaluate the measurement model
        y_check = self.meas.model.evaluate(eval_state)
        # Compute the residual as the difference between the actual measurement
        error = self.meas.value - y_check

        L = self.sqrt_information
        error = L.T @ error

        if compute_jacobians:
            # Jacobians should be a list of length equal to the number of states
            jacobians = [None] * len(states)
            # The Jacobian of the residual is the negative of the measurement
            # model Jacobian
            full_jac = -self.meas.model.jacobian(eval_state)
            # The first 3 columns of the Jacobian are the Jacobian w.r.t the
            # robot state, and the last 2 columns are the Jacobian w.r.t the
            # landmark state
            if compute_jacobians[0]:
                jacobians[0] = L.T @ full_jac[:, :3]
            if compute_jacobians[1]:
                jacobians[1] = L.T @ full_jac[:, 3:]

            return error, jacobians
        return error


def construct_planar_map(x0:SE2State, P0:np.ndarray, input_data:List[VectorInput], process_model:ProcessModel, meas_data=List[Measurement], pose_key_string='x', compute_covariance=True, slam=False, init_landmark:List[StateWithCovariance]=[], use_landmark_prior = False)-> Problem:
    x0_hat = x0.copy()
    x0_hat.state_id = pose_key_string + '0'
    init_pose_est = [x0_hat]
    x = x0_hat.copy()
    for k in range(len(input_data)-1):
        u = input_data[k]
        dt = input_data[k + 1].stamp - u.stamp
        x = process_model.evaluate(x, u, dt)
        x.stamp = x.stamp + dt
        x.state_id = pose_key_string + str(k+1)
        init_pose_est.append(x.copy())
    problem = Problem()
    for i, state in enumerate(init_pose_est):
        problem.add_variable(state.state_id, state)
    
    if slam:
        for i, landmark in enumerate(init_landmark):
            problem.add_variable(landmark.state.state_id, landmark.state.copy())
    
    est_stamps = [state.stamp for state in init_pose_est]

    init_cov = np.copy(P0) # set a small covariance since we've initialized to groundtruth
    prior_residual = PriorResidual(x0_hat.state_id, x0_hat.copy(), init_cov)
    problem.add_residual(prior_residual)
    # SLAM landmark prior
    if slam and use_landmark_prior:
        for i, landmark in enumerate(init_landmark):
            landmark_prior_residual = PriorResidual(landmark.state.state_id, landmark.state.copy(), landmark.covariance)
            problem.add_residual(landmark_prior_residual)
    # Add process residuals
    for k in range(len(input_data) - 1):
        u = input_data[k]

        key_1 = f"{pose_key_string}{k}"
        key_2 = f"{pose_key_string}{k+1}"
        process_residual = ProcessResidual(
            [key_1, key_2],
            process_model,
            u,
        )
        problem.add_residual(process_residual)

    # Before adding in the measurements to the problem, we need to replace the
    # measurement model on the measurements with the measurement model with unknown
    # landmark position
    for k, meas in enumerate(meas_data):
        # Get the pose key
        pose_idx = find_nearest_stamp_idx(est_stamps, meas.stamp)
        # Get state at this id
        pose = init_pose_est[pose_idx]
        key_1 = pose.state_id
        meas_residual = MeasurementResidual(
            [key_1],
            meas,
        )
        if slam:
            landmark_id = meas.model._landmark_id
            meas.model = PointRelativePositionSLAM(pose.state_id, landmark_id, R=meas.model.covariance(None))
            key_2 = landmark_id
            meas_residual = PointRelativePositionResidual([key_1, key_2], meas)
        problem.add_residual(meas_residual)
    
    return problem, init_pose_est


def extract_landmark_est(variables_opt, problem:Problem, init_landmarks:List[VectorState]):
    landmark_results_list: List[StateWithCovariance] = []
    for landmark in init_landmarks:
        landmark_est = variables_opt[landmark.state_id]
        landmark_cov = problem.get_covariance_block(landmark.state_id, landmark.state_id)
        landmark_state_cov = StateWithCovariance(landmark_est, landmark_cov)
        landmark_results_list.append(landmark_state_cov)
    
    return landmark_results_list




    

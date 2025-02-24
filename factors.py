import numpy as np
import scipy.linalg
import navlie as nav
from pymlg.numpy.se2 import SE2, SO2
from typing import Callable, Optional, List
from util.cubatures import gh_cubature, spherical_cubature
from navlie.lib.states import VectorState, SE2State, State, CompositeState
from navlie.types import ProcessModel, Measurement, Input, StateWithCovariance
from navlie.batch.problem import Problem
from navlie.utils import find_nearest_stamp_idx
from util.psd import force_PSD, force_sym
from abc import abstractmethod

class FactoredState:
    def __init__(self, mean:np.ndarray, covariance:np.ndarray, proj_matrix:np.ndarray, stamp:float, cubature_type= 'GH', gh_degree = 3, state_id=None):
        self.mean:np.ndarray = np.copy(mean.reshape((-1,1))).astype(np.float64)
        self.covariance:np.ndarray = np.copy(covariance).astype(np.float64)
        self.sqrt_covariance:np.ndarray = np.linalg.cholesky(covariance)
        self.information:np.ndarray = force_PSD(scipy.linalg.inv(covariance))
        self.projection = np.copy(proj_matrix)
        self.stamp = stamp
        self.dof = np.shape(self.mean)[0]
        self.state_dof = self.dof
        self.expect_scalar:np.ndarray = None
        self.expect_column:np.ndarray = None
        self.expect_matrix:np.ndarray = None
        self.state_id = state_id
        
        # Cubature Method
        if cubature_type=='GH':
            self._gh_degree = gh_degree
            self.cubature:Callable = gh_cubature
            self._unit_sigma_pts, self._weights = self.cubature(order_p=self._gh_degree, state_dof=self.dof)
        elif cubature_type=='spherical':
            self.cubature:Callable = spherical_cubature
            self._unit_sigma_pts, self._weights = self.cubature(order_p=None, state_dof=self.dof)
        else:
            raise NotImplementedError("Implement other cubature methods")
        
        # Generate Sigma Points
        self.generate_new_sigma_pts()

    def generate_new_sigma_pts(self):
        self.sqrt_covariance = np.linalg.cholesky(self.covariance)
        self._sigma_pts = [self.mean + 
                           self.sqrt_covariance @ sp_i.reshape((-1,1)) 
                           for sp_i in self._unit_sigma_pts]
        return
    
    def phi_dx(self):
        return self.information @ self.expect_column

    def phi_dx_dx(self):
        a = self.information @ self.expect_matrix @ self.information
        b = self.information * self.expect_scalar
        return a - b
    
    def phi_dinfo(self):
        a = -0.5 * self.expect_matrix
        b = 0.5 * self.covariance * self.expect_scalar
        c = 0.5 * self.covariance
        # c = np.zeros_like(self.covariance)
        return a + b + c
    
    def compute_expectations(self):
        expect_mu_mu_phi = np.zeros_like(self.information)
        expect_mu_phi = np.zeros_like(self.mean)
        expect_phi = np.zeros((1,1))
        for i, w in enumerate(self._weights):
            phi_k_l = self.eval_phi(self._sigma_pts[i])
            expect_phi += w * phi_k_l
            expect_mu_phi += w * (self._sigma_pts[i] - self.mean) * phi_k_l
            expect_mu_mu_phi += w * (self._sigma_pts[i] - self.mean) @ (self._sigma_pts[i] - self.mean).T  * phi_k_l
        
        self.expect_scalar = np.copy(expect_phi)
        self.expect_column = np.copy(expect_mu_phi)
        self.expect_matrix = np.copy(expect_mu_mu_phi)
        return

    
    def update_factor(self, total_mean, total_information, total_covariance):
        # Project mean, information, covariance
        self.mean = self.projection @ np.copy(total_mean)
        self.information = self.projection @ np.copy(total_information) @ self.projection.T
        # self.information = force_PSD(self.information)
        self.covariance = self.projection @ np.copy(total_covariance) @ self.projection.T
        # self.covariance = force_PSD(self.covariance)
        # Recompute sigma points around new mean / covariance
        self.generate_new_sigma_pts()
        # Recompute expectations using new sigma points
        self.compute_expectations()
        return
    
    # To be implemented by specific type of factors
    @abstractmethod
    def eval_phi(self, sigma_point:np.ndarray) -> np.ndarray:
        pass

    # Get functions
    def get_mean(self):
        return self.mean.copy()
    
    def get_information(self):
        return self.information.copy()
    
    def get_covariance(self):
        return self.covariance.copy()
            
class ProcessFactor(FactoredState):
    def __init__(self, mean, covariance, proj_matrix, stamp, cubature_type='GH', gh_degree=3, state_id=None):
        super().__init__(mean, covariance, proj_matrix, stamp, cubature_type, gh_degree)
        # Individual state dof
        self.state_dof = int(self.dof / 2)

    def link_dependent_state(self, process_model:ProcessModel, u_k_1:Input, x_k:State = None):
        self.process_model:ProcessModel = process_model
        # TODO: Check if need this
        # self.prev_state = prev_state
        self.u = u_k_1
        self.dt = self.stamp - self.u.stamp
        # TODO: Fix the None arguments
        self.Q = process_model.covariance(x_k, self.u, self.dt)
        self.Q_inv = force_PSD(scipy.linalg.inv(self.Q))
        self.compute_expectations()

    def eval_phi(self, sigma_point:np.ndarray):
        # Get sigma points for each associated state
        sp_x_k_1 = sigma_point.reshape((-1,1))[0:self.state_dof]
        sp_x_k = sigma_point.reshape((-1,1))[self.state_dof:]
        # Evaluate process model with sigma x_k_minus_1
        x_k_1 = VectorState(value=sp_x_k_1, stamp=self.u.stamp)
        proc_model_val = self.process_model.evaluate(x_k_1, self.u, self.dt).value.reshape((-1,1))
        # Process factor
        proc_diff = sp_x_k - proc_model_val
        phi = 0.5 * proc_diff.T @ self.Q_inv @ proc_diff
        
        return phi
    
    def get_mean(self):
        return self.mean[self.state_dof:]
    
    def get_information(self):
        return self.information[self.state_dof:, self.state_dof:]
    
    def get_covariance(self):
        return self.covariance[self.state_dof:, self.state_dof:]

class PriorFactor(FactoredState):
    def __init__(self, mean, covariance, proj_matrix, stamp, cubature_type='GH', gh_degree=3):
        super().__init__(mean, covariance, proj_matrix, stamp, cubature_type, gh_degree)
        # Individual state dof
        self.state_dof = self.dof
    
    def link_prior(self, x0:StateWithCovariance):
        # Setup prior values
        self.x0_check = x0.state.value.copy().reshape((-1,1))
        self.P0_check = x0.covariance.copy()
        self.P0_check_inv = force_PSD(scipy.linalg.inv(self.P0_check))
        self.compute_expectations()

    def eval_phi(self, sigma_point:np.ndarray):
        prior_diff = sigma_point.reshape((-1,1)) - self.x0_check
        phi = 0.5 * prior_diff.T @ self.P0_check_inv @ prior_diff

        return phi
    
    def get_mean(self):
        return super().get_mean()
    def get_information(self):
        return super().get_information()
    def get_covariance(self):
        return super().get_covariance()

class MeasurementFactor(FactoredState):
    def __init__(self, mean, covariance, proj_matrix, stamp, cubature_type='GH', gh_degree=3, state_id=None):
        super().__init__(mean, covariance, proj_matrix, stamp, cubature_type, gh_degree, state_id=state_id)
        # Individual state dof
        self.state_dof = self.dof
    
    def link_measurement(self, meas:Measurement):
        self.y_k = meas.value.reshape((-1,1))
        self.meas_model = meas.model
        # TODO: Fix None argument
        self.R_k = np.atleast_2d(meas.model.covariance(x=None))
        self.R_k_inv = force_PSD(scipy.linalg.inv(self.R_k))
        self.compute_expectations()

    def eval_phi(self, sigma_point:np.ndarray):
        # Generate state from sigma point
        sp_state = VectorState(value=sigma_point, stamp=self.stamp)
        # Plug into measurement model
        meas_diff = self.y_k - self.meas_model.evaluate(sp_state).reshape((-1,1))
        # Evaluate prior factor
        phi = 0.5 * meas_diff.T @ self.R_k_inv @ meas_diff
        return phi
    
    def get_mean(self):
        return super().get_mean()
    def get_information(self):
        return super().get_information()
    def get_covariance(self):
        return super().get_covariance()
    
class LandmarkFactor(FactoredState):
    def __init__(self, mean, covariance, proj_matrix, stamp, state_id:str, cubature_type='GH', gh_degree=3):
        super().__init__(mean, covariance, proj_matrix, stamp, cubature_type, gh_degree, state_id=state_id)
        self.state_dof = self.dof

    def eval_phi(self, sigma_point):
        diff = sigma_point.reshape((-1,1)) - self.mean.reshape((-1,1))
        phi = 0.5 * diff.T @ self.information @ diff
        return phi
    
    def get_mean(self):
        return super().get_mean()
    def get_information(self):
        return super().get_information()
    def get_covariance(self):
        return super().get_covariance()

    
class PlanarProcessFactor(ProcessFactor):
    def __init__(self, mean, covariance, proj_matrix, stamp, cubature_type='GH', gh_degree=3):
        super().__init__(mean, covariance, proj_matrix, stamp, cubature_type, gh_degree)
    
    def eval_phi(self, sigma_point):
        to_planar = lambda x: np.vstack((SO2.Log(x.value[0:2, 0:2]), x.value[0:2, 2].reshape((-1,1))))
        # Get sigma points for each associated state
        sp_x_k_1 = sigma_point.reshape((-1,1))[0:self.state_dof]
        sp_x_k = sigma_point.reshape((-1,1))[self.state_dof:]
        # Evaluate process model with sigma x_k_minus_1
        x_k_1_val = SE2.from_components(C=sp_x_k_1[0], r=sp_x_k_1[1:])
        x_k_val = SE2.from_components(C=sp_x_k[0], r=sp_x_k[1:])
        x_k_1 = SE2State(value=x_k_1_val, stamp=self.u.stamp)
        x_k = SE2State(value=x_k_val, stamp=self.stamp)
        proc_model_state = self.process_model.evaluate(x_k_1, self.u, self.dt)
        proc_diff = x_k.minus(proc_model_state).reshape((-1,1))
        # proc_diff = sp_x_k - to_planar(proc_model_state)
        phi = 0.5 * proc_diff.T @ self.Q_inv @ proc_diff
        return phi

class PlanarMeasurementFactor(MeasurementFactor):
    def __init__(self, mean, covariance, proj_matrix, stamp, cubature_type='GH', gh_degree=3):
        super().__init__(mean, covariance, proj_matrix, stamp, cubature_type, gh_degree)
    def eval_phi(self, sigma_point):
        # Create SE2 State from sigma points
        sp_x_k = sigma_point.reshape((-1,1))
        sp_se2 = SE2.from_components(C=SO2.Exp(sp_x_k[0]), r=sp_x_k[1:])
        sp_state = SE2State(value=sp_se2, stamp=self.stamp)
        meas_diff = self.y_k - self.meas_model.evaluate(sp_state).reshape((-1,1))
        phi = 0.5 * meas_diff.T @ self.R_k_inv @ meas_diff
        return phi
    

class SLAMFactor(MeasurementFactor):
    def __init__(self, mean, covariance, proj_matrix, stamp, cubature_type='GH', gh_degree=3):
        super().__init__(mean, covariance, proj_matrix, stamp, cubature_type, gh_degree)
        self.state_dof = self.dof
        self.landmark_dof = None
    
    def link_landmark(self, landmark_factor:LandmarkFactor):
        self.landmark_factor = landmark_factor
        self.landmark_dof = landmark_factor.dof
        self.projection = np.block([[self.projection], [landmark_factor.projection]])
    
    def eval_phi(self, state_sp:np.ndarray, landmark_sp:np.ndarray):
        # Create SE2 State from sigma point
        sp_x_k = state_sp.reshape((-1,1))
        sp_se2 = SE2.from_components(C=SO2.Exp(sp_x_k[0]), r=sp_x_k[1:])
        se2_state = SE2State(value=sp_se2, stamp=self.stamp, state_id=self.state_id)
        # Create landmark state from sigma point
        landmark_state = VectorState(value=landmark_sp, stamp=self.stamp, state_id=self.landmark_factor.state_id)
        # Combine into composite state
        slam_state = CompositeState(state_list=[se2_state, landmark_state], stamp=self.stamp)

        # Evaluate Measurement Model
        meas_diff = self.y_k - self.meas_model.evaluate(slam_state).reshape((-1,1))
        # Evaluate prior factor
        phi = 0.5 * meas_diff.T @ self.R_k_inv @ meas_diff
        return phi
    
    def compute_expectations(self):
        # TODO: Fix this for 
        expect_mu_mu_phi = np.zeros((self.state_dof+self.landmark_dof, self.state_dof+self.landmark_dof))
        expect_mu_phi = np.zeros((self.state_dof+self.landmark_dof,1))
        expect_phi = np.zeros((1,1))
        for i, w_state in enumerate(self._weights):
            for j, w_land in enumerate(self.landmark_factor._sigma_pts):
                phi_k_l = self.eval_phi(self._sigma_pts[i], self.landmark_factor._sigma_pts[j])
                # TODO: Figure out how to combine this
                # I'm thinking you'd need to stack in order to send information back to the landmark state, in order to force landmark state to agree with all measurements.
                # Or maybe don't stack and just have them within the landmark factor itself
                stack_sp = np.vstack((self._sigma_pts[i].reshape((-1,1)), self.landmark_factor._sigma_pts[j].reshape((-1,1))))
                stack_mean = np.vstack((self.mean, self.landmark_factor.mean))
                expect_phi += w_state * w_land * phi_k_l
                expect_mu_phi += w_state * w_land * (stack_sp - stack_mean) * phi_k_l
                expect_mu_mu_phi += w_state * (stack_sp - stack_mean) @ (stack_sp - stack_mean).T  * phi_k_l
        
        self.expect_scalar = np.copy(expect_phi)
        self.expect_column = np.copy(expect_mu_phi)
        self.expect_matrix = np.copy(expect_mu_mu_phi)
        return
        



def construct_factor_list(x0:StateWithCovariance, input_data:List[Input], meas_data:List[Measurement], proc_model:ProcessModel, cubature_type ='GH', gh_deg = 3):
    factored_state_list = []
    factored_stamp_list = []
    state_dof = x0.state.dof

    # Define prior
    proj_0 = np.zeros((state_dof, len(input_data)*state_dof))
    proj_0[:, :state_dof] = np.eye(state_dof)
    prior_factor = PriorFactor(mean=x0.state.value, covariance=x0.covariance, proj_matrix=proj_0, stamp=x0.state.stamp, gh_degree=gh_deg, cubature_type=cubature_type)
    prior_factor.link_prior(x0=x0)
    factored_state_list.append(prior_factor)
    factored_stamp_list.append(x0.state.stamp)

    # Define process prior
    proj_proc_empty = np.zeros((2*state_dof, len(input_data)*state_dof))
    proj_idx = 0
    x_k_1 = x0.state.copy()
    P_k_1 = force_sym(x0.covariance)
    for i in range(len(input_data)-1):
        u_k_1:Input = input_data[i]
        u_k:Input = input_data[i+1]
        dt_u = u_k.stamp - u_k_1.stamp
        x_k = proc_model.evaluate(x_k_1, u=u_k_1, dt=dt_u)
        A_k = proc_model.jacobian(x_k_1, u=u_k_1, dt=dt_u)
        Q_k = proc_model.covariance(x=x_k_1, u=u_k_1, dt=dt_u)
        P_k = A_k @ P_k_1 @ A_k.T + Q_k
        P_k = force_sym(P_k)
        p_mean = np.vstack((x_k_1.value.reshape((-1,1)), x_k.value.reshape((-1,1))))
        if i==0:
            P_covar = np.block([[P_k_1, np.zeros((2,2))],
                                [np.zeros((2,2)), P_k]])
        else:
            P_covar = np.block([[P_k_1, np.zeros((2,2))],
                                [np.zeros((2,2)), P_k]])
            # P_covar = np.block([[P_k_1, P_k_1 @ A_k.T],
            #                     [A_k @ P_k_1, P_k]])
        P_covar = force_sym(P_covar)
        proj_k = proj_proc_empty.copy()
        proj_k[:, proj_idx:proj_idx+(2*state_dof)] = np.eye(2*state_dof)
        p_fac_k = ProcessFactor(mean=p_mean, covariance=P_covar, proj_matrix=proj_k, stamp=u_k.stamp, gh_degree=gh_deg, cubature_type=cubature_type)
        p_fac_k.link_dependent_state(process_model=proc_model, u_k_1 = u_k_1)
        factored_state_list.append(p_fac_k)
        factored_stamp_list.append(u_k.stamp)
        proj_idx += state_dof
        x_k_1 = x_k.copy()
        P_k_1 = P_k.copy()

    # Define measurement factors
    proj_meas_empty = np.zeros((state_dof, len(input_data)*state_dof))
    for i in range(len(meas_data)):
        meas:Measurement = meas_data[i]
        # TODO: Fix assumption that measurement is synchronized with inputs
        if meas.stamp<=input_data[-1].stamp:
            x_k_idx = factored_stamp_list.index(meas.stamp)
            x_k:FactoredState = factored_state_list[x_k_idx]
            
            proj_k = proj_meas_empty.copy()
            if isinstance(x_k, PriorFactor):
                proj_k = x_k.projection.copy()
                # x_k.link_measurement(meas=meas)
            else:
                x_k:ProcessFactor
                proj_k = x_k.projection[x_k.state_dof:, :].copy()
                # x_k.link_measurement(meas=meas)
            meas_factor = MeasurementFactor(mean=x_k.get_mean(), covariance=x_k.get_covariance(), stamp=meas.stamp, proj_matrix=proj_k, gh_degree=gh_deg, cubature_type=cubature_type)
            meas_factor.link_measurement(meas=meas)
            factored_state_list.append(meas_factor)
    
    return factored_state_list

def construct_planar_factor_list(x0:SE2State, P0:np.ndarray, input_data:List[Input], meas_data:List[Measurement], proc_model:ProcessModel, cubature_type ='GH', gh_deg = 3):
    to_planar = lambda x: np.vstack((SO2.Log(x.value[0:2, 0:2]), x.value[0:2, 2].reshape((-1,1)))) #x.group.Log(x.value)
    factored_state_list = []
    factored_stamp_list = []
    state_dof = x0.dof

    # Define prior
    proj_0 = np.zeros((state_dof, len(input_data)*state_dof))
    proj_0[:, :state_dof] = np.eye(state_dof)
    
    prior_factor = PriorFactor(mean=to_planar(x0), covariance=P0, proj_matrix=proj_0, stamp=x0.stamp, gh_degree=gh_deg, cubature_type=cubature_type)
    x0_state_covar = StateWithCovariance(VectorState(value=to_planar(x0)), covariance=P0)
    prior_factor.link_prior(x0=x0_state_covar)
    factored_state_list.append(prior_factor)
    factored_stamp_list.append(x0.stamp)

    # Define process prior
    proj_proc_empty = np.zeros((2*state_dof, len(input_data)*state_dof))
    proj_idx = 0
    x_k_1 = x0.copy()
    P_k_1 = force_sym(P0)
    for i in range(len(input_data)-1):
        u_k_1:Input = input_data[i]
        u_k:Input = input_data[i+1]
        dt_u = u_k.stamp - u_k_1.stamp
        x_k = proc_model.evaluate(x_k_1, u=u_k_1, dt=dt_u)
        A_k = proc_model.jacobian(x_k_1, u=u_k_1, dt=dt_u)
        Q_k = proc_model.covariance(x=x_k_1, u=u_k_1, dt=dt_u)
        P_k = A_k @ P_k_1 @ A_k.T + Q_k
        P_k = force_sym(P_k)
        x_k_1_p = to_planar(x_k_1)
        x_k_p = to_planar(x_k)
        p_mean = np.vstack((x_k_1_p.reshape((-1,1)), x_k_p.reshape((-1,1))))
        if i==0:
            P_covar = np.block([[P_k_1, np.zeros_like(P_k)],
                                [np.zeros_like(P_k), P_k]])
        else:
            P_covar = np.block([[P_k_1, np.zeros_like(P_k)],
                                [np.zeros_like(P_k), P_k]])
            # P_covar = np.block([[P_k_1, P_k_1 @ A_k.T],
            #                     [A_k @ P_k_1, P_k]])
        P_covar = force_sym(P_covar)
        proj_k = proj_proc_empty.copy()
        proj_k[:, proj_idx:proj_idx+(2*state_dof)] = np.eye(2*state_dof)
        p_fac_k = PlanarProcessFactor(mean=p_mean, covariance=P_covar, proj_matrix=proj_k, stamp=u_k.stamp, gh_degree=gh_deg, cubature_type=cubature_type)
        p_fac_k.link_dependent_state(process_model=proc_model, u_k_1 = u_k_1, x_k=x_k_1)
        factored_state_list.append(p_fac_k)
        factored_stamp_list.append(u_k.stamp)
        proj_idx += state_dof
        x_k_1 = x_k.copy()
        P_k_1 = P_k.copy()

    # Define measurement factors
    proj_meas_empty = np.zeros((state_dof, len(input_data)*state_dof))
    for i in range(len(meas_data)):
        meas:Measurement = meas_data[i]
        # TODO: Fix assumption that measurement is synchronized with inputs
        if meas.stamp<=input_data[-1].stamp:
            x_k_idx = factored_stamp_list.index(meas.stamp)
            x_k:FactoredState = factored_state_list[x_k_idx]
            
            proj_k = proj_meas_empty.copy()
            if isinstance(x_k, PriorFactor):
                proj_k = x_k.projection.copy()
                # x_k.link_measurement(meas=meas)
            else:
                x_k:ProcessFactor
                proj_k = x_k.projection[x_k.state_dof:, :].copy()
                # x_k.link_measurement(meas=meas)
            meas_factor = PlanarMeasurementFactor(mean=x_k.get_mean(), covariance=x_k.get_covariance(), stamp=meas.stamp, proj_matrix=proj_k, gh_degree=gh_deg, cubature_type=cubature_type)
            meas_factor.link_measurement(meas=meas)
            factored_state_list.append(meas_factor)
    
    return factored_state_list
        
def construct_from_map(opt_variables, problem:Problem, input_data:List[nav.types.Input], meas_data:List[Measurement], proc_model:ProcessModel, cubature_type ='GH', gh_deg = 3) -> List[FactoredState]:
    to_planar = lambda x: np.vstack((SO2.Log(x.value[0:2, 0:2]), x.value[0:2, 2].reshape((-1,1)))) #x.group.Log(x.value)
    factored_state_list = []
    factored_stamp_list = []

    # Define prior
    x0_state:State = opt_variables['x0']
    state_dof = x0_state.dof
    P0 = problem.get_covariance_block(x0_state.state_id, x0_state.state_id)
    
    proj_0 = np.zeros((state_dof, len(input_data)*state_dof))
    proj_0[:, :state_dof] = np.eye(state_dof)

    prior_factor = PriorFactor(mean=to_planar(x0_state.copy()), covariance=P0, proj_matrix=proj_0, stamp=x0_state.stamp, gh_degree=gh_deg, cubature_type=cubature_type)
    x0_state_covar = StateWithCovariance(VectorState(value=to_planar(x0_state.copy()), stamp=x0_state.stamp), covariance=P0)
    prior_factor.link_prior(x0=x0_state_covar)
    factored_state_list.append(prior_factor)
    factored_stamp_list.append(x0_state.stamp)

    # Define process prior
    proj_proc_empty = np.zeros((2*state_dof, len(input_data)*state_dof))
    proj_idx = 0
    for i in range(len(input_data)-1):
        u_k_1:Input = input_data[i]
        u_k:Input = input_data[i+1]
        dt_u = u_k.stamp - u_k_1.stamp
        if i+1 == 70:
            pass
        x_k_1:State = opt_variables['x' + str(i)].copy()
        P_k_1 = problem.get_covariance_block(x_k_1.state_id, x_k_1.state_id)
        x_k:State = opt_variables['x'+str(i+1)].copy()
        P_k = problem.get_covariance_block(x_k.state_id, x_k.state_id)
        P_joint = problem.get_covariance_block(x_k.state_id, x_k_1.state_id)
        x_k_1_p = to_planar(x_k_1)
        x_k_p = to_planar(x_k)
        p_mean = np.vstack((x_k_1_p.reshape((-1,1)), x_k_p.reshape((-1,1))))
        P_covar = np.block([[P_k_1, np.zeros((3,3))],
                            [np.zeros((3,3)), P_k]])
        P_covar = force_sym(P_covar)
        proj_k = proj_proc_empty.copy()
        proj_k[:, proj_idx:proj_idx+(2*state_dof)] = np.eye(2*state_dof)
        p_fac_k = PlanarProcessFactor(mean=p_mean, covariance=P_covar, proj_matrix=proj_k, stamp=u_k.stamp, gh_degree=gh_deg, cubature_type=cubature_type)
        p_fac_k.link_dependent_state(process_model=proc_model, u_k_1 = u_k_1, x_k=x_k_1)
        factored_state_list.append(p_fac_k)
        factored_stamp_list.append(u_k.stamp)
        proj_idx += state_dof

    # Define measurement factors
    proj_meas_empty = np.zeros((state_dof, len(input_data)*state_dof))
    for i in range(len(meas_data)):
        meas:Measurement = meas_data[i]
        if meas.stamp<=input_data[-1].stamp:
            x_k_idx = find_nearest_stamp_idx(factored_stamp_list, meas.stamp)
            x_k:FactoredState = factored_state_list[x_k_idx]
            proj_k = proj_meas_empty.copy()
            if isinstance(x_k, PriorFactor):
                proj_k = x_k.projection.copy()
                # x_k.link_measurement(meas=meas)
            else:
                x_k:ProcessFactor
                proj_k = x_k.projection[x_k.state_dof:, :].copy()
                # x_k.link_measurement(meas=meas)
            meas_factor = PlanarMeasurementFactor(mean=x_k.get_mean(), covariance=x_k.get_covariance(), stamp=meas.stamp, proj_matrix=proj_k, gh_degree=gh_deg, cubature_type=cubature_type)
            meas_factor.link_measurement(meas=meas)
            factored_state_list.append(meas_factor)

    return factored_state_list








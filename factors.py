import numpy as np
import scipy.linalg
import navlie as nav
from typing import Callable, Optional, List
from util.cubatures import gh_cubature, spherical_cubature
from navlie.lib.states import VectorState
from navlie.types import ProcessModel, Measurement, Input, StateWithCovariance
from util.psd import force_PSD, force_sym
from abc import abstractmethod

class FactoredState:
    def __init__(self, mean:np.ndarray, covariance:np.ndarray, proj_matrix:np.ndarray, stamp:float, cubature_type= 'GH', gh_degree = 3):
        self.mean:np.ndarray = mean.reshape((-1,1))
        self.covariance:np.ndarray = covariance.copy()
        self.sqrt_covariance:np.ndarray = np.linalg.cholesky(covariance)
        self.information:np.ndarray = force_PSD(scipy.linalg.inv(covariance))
        self.projection = proj_matrix.copy()
        self.stamp = stamp
        self.dof = np.shape(self.mean)[0]
        self.state_dof = self.dof
        self.expect_scalar:np.ndarray = None
        self.expect_column:np.ndarray = None
        self.expect_matrix:np.ndarray = None
        
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
        
        # Generate Unit Sigma Points
        self.generate_new_sigma_pts()

    def generate_new_sigma_pts(self):
        self._sqrt_cov = np.linalg.cholesky(self.covariance)
        self._sigma_pts = [self.mean + self._sqrt_cov @ sp_i.reshape((-1,1)) 
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
        
        self.expect_scalar = expect_phi.copy()
        self.expect_column = expect_mu_phi.copy()
        self.expect_matrix = expect_mu_mu_phi.copy()
        return

    
    def update_factor(self, total_mean, total_information, total_covariance):
        # Project mean, information, covariance
        self.mean = self.projection @ total_mean
        self.information = self.projection @ total_information @ self.projection.T
        # self.information = force_PSD(self.information)
        self.covariance = self.projection @ total_covariance @ self.projection.T
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
    def __init__(self, mean, covariance, proj_matrix, stamp, cubature_type='GH', gh_degree=3):
        super().__init__(mean, covariance, proj_matrix, stamp, cubature_type, gh_degree)
        # Individual state dof
        self.state_dof = int(self.dof / 2)

    def link_dependent_state(self, process_model:ProcessModel, u_k_1:Input):
        self.process_model = process_model
        # TODO: Check if need this
        # self.prev_state = prev_state
        self.u = u_k_1
        self.dt = self.stamp - self.u.stamp
        # TODO: Fix the None arguments
        self.Q = process_model.covariance(None, None, self.dt)
        self.Q_inv = scipy.linalg.inv(self.Q)
        self.compute_expectations()

    def eval_phi(self, sigma_point:np.ndarray):
        sp_x_k_1 = sigma_point.reshape((-1,1))[0:self.state_dof]
        sp_x_k = sigma_point.reshape((-1,1))[self.state_dof:]
        x_k_1 = VectorState(value=sp_x_k_1, stamp=self.u.stamp)
        proc_model_val = self.process_model.evaluate(x_k_1, self.u, self.dt).value.reshape((-1,1))
        phi = 0.5 * (sp_x_k - proc_model_val).T @ self.Q_inv @ (sp_x_k - proc_model_val)
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
        self.x0_check = x0.state.value.copy().reshape((-1,1))
        self.P0_check = x0.covariance.copy()
        self.P0_check_inv = scipy.linalg.inv(self.P0_check)
        self.compute_expectations()

    def eval_phi(self, sigma_point:np.ndarray):
        phi = 0.5 * (sigma_point.reshape((-1,1)) - self.x0_check).T @ self.P0_check_inv @ (sigma_point.reshape((-1,1)) - self.x0_check)
        return phi
    
    def get_mean(self):
        return super().get_mean()
    def get_information(self):
        return super().get_information()
    def get_covariance(self):
        return super().get_covariance()

class MeasurementFactor(FactoredState):
    def __init__(self, mean, covariance, proj_matrix, stamp, cubature_type='GH', gh_degree=3):
        super().__init__(mean, covariance, proj_matrix, stamp, cubature_type, gh_degree)
        # Individual state dof
        self.state_dof = self.dof
    
    def link_measurement(self, meas:Measurement):
        self.y_k = meas.value.reshape((-1,1))
        self.meas_model = meas.model
        # TODO: Fix None argument
        self.R_k = np.atleast_2d(meas.model.covariance(x=None))
        self.R_k_inv = scipy.linalg.inv(self.R_k)
        self.compute_expectations()

    def eval_phi(self, sigma_point:np.ndarray):
        sp_state = VectorState(value=sigma_point, stamp=self.stamp)
        # phi = 0.5 * (self.y_k - self.meas_model.evaluate(sp_state).reshape((-1,1))).T @ self.R_k_inv @ (self.y_k - self.meas_model.evaluate(sp_state).reshape((-1,1)))
        phi = 0.5 * (self.y_k.ravel() - self.meas_model.evaluate(sp_state).ravel())**2 / self.R_k
        return phi
    
    def get_mean(self):
        return super().get_mean()
    def get_information(self):
        return super().get_information()
    def get_covariance(self):
        return super().get_covariance()
    
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
    P_k_1 = x0.covariance
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
    proj_idx = 0
    for i in range(len(meas_data)):
        meas:Measurement = meas_data[i]
        # TODO: Fix assumption that measurement is synchronized with inputs
        if meas.stamp<=input_data[-1].stamp:
            x_k_idx = factored_stamp_list.index(meas.stamp)
            x_k:FactoredState = factored_state_list[x_k_idx]
            
            proj_k = proj_meas_empty.copy()
            if isinstance(x_k, PriorFactor):
                proj_k = x_k.projection.copy()
            else:
                proj_k = x_k.projection[x_k.state_dof:, :].copy()
            meas_factor = MeasurementFactor(mean=x_k.get_mean(), covariance=x_k.get_covariance(), stamp=meas.stamp, proj_matrix=proj_k, gh_degree=gh_deg, cubature_type=cubature_type)
            meas_factor.link_measurement(meas=meas)
            factored_state_list.append(meas_factor)
            
        proj_idx += state_dof

    return factored_state_list

def limit_data(input_data:List[Input], meas_data:List[Measurement], proc_model:ProcessModel, max_length=100):
    meas_stamps = [y.stamp for y in meas_data]
    if len(input_data) > max_length:
        input_data_lim = input_data[:max_length]
        last_u = input_data_lim[-1]
        last_y_idx = nav.find_nearest_stamp_idx(meas_stamps, last_u.stamp)
        meas_data_lim = meas_data[:last_y_idx]
        return input_data_lim, meas_data_lim
        
    


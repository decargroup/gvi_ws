import numpy as np
import scipy.linalg
import navlie as nav
from typing import Callable, Optional, List, Tuple
from util.cubatures import gh_cubature, spherical_cubature
from navlie.lib.states import VectorState, State, MatrixLieGroupState, MatrixLieGroup
from navlie.types import ProcessModel, Measurement, Input, StateWithCovariance
from util.psd import force_PSD, force_sym
from abc import abstractmethod

class FactoredMLGState:
    def __init__(self,mean:State, covariance:np.ndarray, proj_matrix:np.ndarray, stamp:float, cubature_type='GH', gh_degree = 3):
        self.mean = mean.copy()
        # self.mean.value = np.copy(mean.value).astype(np.float64)
        if isinstance(mean, MatrixLieGroupState):
            self.group = mean.group
        else:
            self.group = None
        self.covariance:np.ndarray = np.copy(covariance).astype(np.float64)
        self.total_covariance = np.copy(self.covariance)
        self.sqrt_covariance:np.ndarray = np.linalg.cholesky(covariance)
        
        self.information:np.ndarray = force_PSD(scipy.linalg.inv(covariance))
        self.total_information = np.copy(self.information)
        self.projection = np.copy(proj_matrix)
        self.total_projection = np.copy(proj_matrix)
        self.stamp = stamp
        self.dof = self.mean.dof
        self.total_dof = self.mean.dof
        self.expect_scalar:np.ndarray = None
        self.expect_column:np.ndarray = None
        self.expect_matrix:np.ndarray = None
        self.sigma_pts:List[MatrixLieGroupState] = None

        # Cubature Method, and Unit Sigma Points
        self.generate_unit_sigma_pts(cubature_type=cubature_type, order=gh_degree)
        self.generate_new_sigma_pts()
        return
    
    def generate_unit_sigma_pts(self, cubature_type, order = None):
        if cubature_type=='GH':
            self._gh_degree = order
            self.cubature:Callable = gh_cubature
            self._unit_sigma_pts, self.weights = self.cubature(order_p=self._gh_degree, state_dof=self.total_dof)
        elif cubature_type=='spherical':
            self.cubature:Callable = spherical_cubature
            self._gh_degree = order
            self._unit_sigma_pts, self.weights = self.cubature(order_p=None, state_dof=self.total_dof)
        else:
            raise NotImplementedError("Implement other cubature methods")


    def generate_new_sigma_pts(self):
        self.sqrt_covariance = np.linalg.cholesky(self.covariance)
        # Now Sigma Points will be on Lie Group
        # So will also be list of MatrixLieGroup
        self.sigma_pts = [self.mean.plus(self.sqrt_covariance @ sp_i.reshape((-1,1)))
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
        expect_mu_phi = np.zeros((self.dof, 1), dtype=np.float64)
        expect_phi = np.zeros((1,1))
        for i, w in enumerate(self.weights):
            phi_k_l = self.eval_phi(self.sigma_pts[i])
            expect_phi += w * phi_k_l
            expect_mu_phi += w * (self.sigma_pts[i].minus(self.mean).reshape((self.dof, 1))) * phi_k_l
            expect_mu_mu_phi += w * (self.sigma_pts[i].minus(self.mean).reshape((self.dof, 1))) @ (self.sigma_pts[i].minus(self.mean).reshape((self.dof, 1))).T  * phi_k_l
        
        self.expect_scalar = expect_phi.copy()
        self.expect_column = expect_mu_phi.copy()
        self.expect_matrix = expect_mu_mu_phi.copy()

        return
    
    @abstractmethod
    def eval_phi(self, sigma_point:MatrixLieGroupState) -> np.ndarray:
        pass

    @abstractmethod
    def update_factor(self, total_mean, total_information, total_covariance):
        pass

    def update_state(self, total_mean, total_information, total_covariance):
        # Project mean, information, covariance
        # This is in the vector space?
        # TODO: Try just exp rather than Exp
        mean_vector = self.projection @ np.copy(total_mean)
        if self.group is not None:
            # self.mean.value = self.group.exp(mean_vector)
            self.mean.value = np.copy(self.group.Exp(mean_vector))
        else:
            self.mean.value = np.copy(mean_vector)
        
        self.information = self.projection @ np.copy(total_information) @ self.projection.T
        self.covariance = self.projection @ np.copy(total_covariance) @ self.projection.T
        # self.covariance = force_PSD(self.covariance)
        # Recompute sigma points around new mean / covariance
        self.generate_new_sigma_pts()
        # Recompute expectations using new sigma points
        self.compute_expectations()
        return

    # Get functions
    def get_mean(self):
        """
        Return a copy of the mean state.
        """
        return self.mean.copy()
    
    def get_information(self):
        return self.information.copy()
    
    def get_covariance(self):
        return self.covariance.copy()
    
    def get_mean_vector(self):
        """
        Returns copy of the mean state, in a vector representation. 
        """
        if self.group is not None:
            return self.group.Log(self.mean.value).reshape((self.dof, 1)).copy()
        else:
            return self.mean.value.reshape((self.dof, 1)).copy()
        

class ProcessFactor(FactoredMLGState):
    def __init__(self, mean:State, covariance:np.ndarray, proj_matrix:np.ndarray, related_factor:FactoredMLGState, process_model:ProcessModel, u:Input, stamp, cubature_type='GH', gh_degree=3):
        # State level fields
        self.mean = mean.copy()
        # self.mean.value = np.copy(mean.value).astype(np.float64)
        if isinstance(mean, MatrixLieGroupState):
            self.group = mean.group
        else:
            self.group = None

        self.covariance:np.ndarray = np.copy(covariance).astype(np.float64)
        self.information:np.ndarray = force_PSD(scipy.linalg.inv(covariance))
        self.total_information = np.copy(self.information)
        self.projection = np.copy(proj_matrix)
        self.stamp = stamp
        self.dof = self.mean.dof
        self.expect_scalar:np.ndarray = None
        self.expect_column:np.ndarray = None
        self.expect_matrix:np.ndarray = None
        # sigma points have (related factor pts, cur factor pts)
        self.sigma_pts:List[Tuple[State]] = None
        
        # Factor level fields, with total values dependent on the related factor.
        self.related_factor = related_factor
        self.total_dof = self.dof + self.related_factor.dof
        self.process_model = process_model
        self.u = u
        self.dt = self.stamp - self.u.stamp
        self._Q = self.process_model.covariance(self.mean, self.u, self.dt)
        self._Q_inv = force_PSD(scipy.linalg.inv(self._Q))
        
        # Combine Covariance and Information sizing
        zero_block = np.zeros((self.related_factor.dof, self.dof))
        self.total_covariance = np.block([[self.related_factor.covariance, zero_block], [zero_block.T, self.covariance]])
        self.total_information = force_PSD(scipy.linalg.inv(self.total_covariance))
        self.total_projection = np.vstack((self.related_factor.projection, self.projection))

        # Cubature Method, and Unit Sigma Points
        self.generate_unit_sigma_pts(cubature_type=cubature_type, order=gh_degree)
        self.generate_new_sigma_pts()
        # Compute expectations using the Sigma Points
        self.compute_expectations()
        return
    
    def generate_new_sigma_pts(self):
        self.sqrt_covariance = np.linalg.cholesky(self.total_covariance)
        vector_sigma_points = [self.sqrt_covariance @ sp_i.reshape((-1,1)) for sp_i in self._unit_sigma_pts]
        self.sigma_pts = []
        for sp_vec in vector_sigma_points:
            sp_vec_prev = sp_vec[0:self.related_factor.dof]
            sp_vec_cur = sp_vec[self.dof:]
            sp_lie_prev = self.related_factor.mean.plus(sp_vec_prev)
            sp_lie_cur = self.mean.plus(sp_vec_cur)
            self.sigma_pts.append((sp_lie_prev, sp_lie_cur))

    
    def phi_dx(self):
        return self.total_information @ self.expect_column

    def phi_dx_dx(self):
        a = self.total_information @ self.expect_matrix @ self.total_information
        b = self.total_information * self.expect_scalar
        return a - b
    
    def eval_phi(self, cur_sigma_point:State, prev_sigma_point:State):
        propagated = self.process_model.evaluate(prev_sigma_point, self.u, self.dt)
        diff = cur_sigma_point.minus(propagated).reshape((self.dof, 1))
        phi_proc = 0.5 * diff.T @ self._Q_inv @ diff
        return phi_proc
    
    def compute_expectations(self):
        total_dim = self.total_dof
        expect_mu_mu_phi = np.zeros((total_dim, total_dim))
        expect_mu_phi = np.zeros((total_dim, 1))
        expect_phi = np.zeros((1,1))
        for i, w in enumerate(self.weights):
            prev_sp = self.sigma_pts[i][0]
            cur_sp = self.sigma_pts[i][1]
            phi_k_l = self.eval_phi(cur_sigma_point=cur_sp, prev_sigma_point=prev_sp)
            expect_phi += w * phi_k_l
            prev_diff = prev_sp.minus(self.related_factor.mean).reshape((self.related_factor.dof, 1))
            cur_diff = cur_sp.minus(self.mean).reshape((self.dof, 1))
            diff = np.vstack((prev_diff, cur_diff))
            expect_mu_phi += w * phi_k_l * diff
            expect_mu_mu_phi += w * phi_k_l * (diff @ diff.T)

        self.expect_scalar = np.copy(expect_phi)
        self.expect_column = np.copy(expect_mu_phi)
        self.expect_matrix = np.copy(expect_mu_mu_phi)
        return 
    
    def update_factor(self, total_mean, total_information, total_covariance):
        self.total_information = self.total_projection @ np.copy(total_information) @ self.total_projection.T
        self.total_covariance = self.total_projection @ np.copy(total_covariance) @ self.total_projection.T

        self.update_state(total_mean, total_information, total_covariance)
        
        # self.related_factor.update_state(total_mean, total_information, total_covariance)
        # Recompute sigma points around new mean / covariance
        self.generate_new_sigma_pts()
        # Recompute expectations using new sigma points
        self.compute_expectations()

    def get_cross_information(self):
        return self.total_information[0:self.related_factor.dof, self.dof:]
        
    
class PriorFactor(FactoredMLGState):
    def __init__(self, mean, covariance, proj_matrix, prior:StateWithCovariance, stamp, cubature_type='GH', gh_degree=3):
        super().__init__(mean, covariance, proj_matrix, stamp, cubature_type, gh_degree)  
        # Setup prior terms
        self.x0_check = prior.state.copy()
        self.x0_check.value = np.copy(prior.state.value)
        self.P0_check = np.copy(prior.covariance)
        self.P0_check_inv = force_PSD(scipy.linalg.inv(self.P0_check))
        # Compute relevant expectations
        self.compute_expectations()

    def eval_phi(self, sigma_point):
        prior_diff = sigma_point.minus(self.mean).reshape((self.dof, 1))
        phi_prior = 0.5 * prior_diff.T @ self.P0_check_inv @ prior_diff
        return phi_prior
    
    def update_factor(self, total_mean, total_information, total_covariance):
        # Doesn't have any dependence on another factor, so can just update the individual state
        return super().update_state(total_mean, total_information, total_covariance)
    
    
class MeasurementFactor(FactoredMLGState):
    def __init__(self, mean, covariance, proj_matrix, meas:Measurement, stamp, cubature_type='GH', gh_degree=3):
        super().__init__(mean, covariance, proj_matrix, stamp, cubature_type, gh_degree) 
        self.y_k = np.copy(meas.value)
        self.meas_model = meas.model
        self.R_k = np.atleast_2d(meas.model.covariance(self.mean))
        self.R_k_inv = force_PSD(scipy.linalg.inv(self.R_k))
        self.compute_expectations()

    def eval_phi(self, sigma_point):
        # Calculate measurement model propagated difference
        meas_diff = self.y_k - self.meas_model.evaluate(sigma_point)
        phi_meas = 0.5 * meas_diff.T @ self.R_k_inv @ meas_diff
        return phi_meas
    
    def update_factor(self, total_mean, total_information, total_covariance):
        # Doesn't have any dependence on another factor, so can just update the individual state
        return super().update_state(total_mean, total_information, total_covariance)

def construct_factor_list(x0:StateWithCovariance, input_data:List[Input], meas_data:List[Measurement], proc_model:ProcessModel, cubature_type ='GH', gh_deg = 3):
    factored_state_list = []
    factored_stamp_list = []
    state_dof = x0.state.dof

    # Define prior
    proj_0 = np.zeros((state_dof, len(input_data)*state_dof))
    proj_0[:, :state_dof] = np.eye(state_dof)
    prior_factor = PriorFactor(mean=x0.state.copy(), covariance=x0.covariance, proj_matrix=proj_0, prior=x0.copy(), stamp=x0.state.stamp, gh_degree=gh_deg, cubature_type=cubature_type)
    factored_state_list.append(prior_factor)
    factored_stamp_list.append(x0.state.stamp)

    # Define process prior
    proj_idx = state_dof
    x_k_1 = x0.state.copy()
    P_k_1 = force_sym(x0.covariance)
    for i in range(len(input_data)-1):
        u_k_1:Input = input_data[i]
        u_k:Input = input_data[i+1]
        dt_u = u_k.stamp - u_k_1.stamp
        x_k:State = proc_model.evaluate(x_k_1, u=u_k_1, dt=dt_u)
        x_k.stamp = u_k.stamp
        A_k = proc_model.jacobian(x_k_1, u=u_k_1, dt=dt_u)
        Q_k = proc_model.covariance(x=x_k_1, u=u_k_1, dt=dt_u)
        P_k = A_k @ P_k_1 @ A_k.T + Q_k
        P_k = force_sym(P_k)
        proj_k = np.zeros((state_dof, len(input_data)*state_dof))
        proj_k[:, proj_idx:proj_idx+state_dof] = np.eye(state_dof)
        proc_factor_k = ProcessFactor(mean=x_k, covariance=P_k, proj_matrix=proj_k, related_factor=factored_state_list[-1], process_model=proc_model, u=u_k_1, stamp=u_k.stamp, cubature_type=cubature_type, gh_degree=gh_deg)
        factored_state_list.append(proc_factor_k)
        factored_stamp_list.append(u_k.stamp)
        proj_idx += state_dof
        x_k_1 = x_k.copy()
        x_k_1.value = np.copy(x_k.value)
        P_k_1 = np.copy(P_k)
    
    # Define measurement factors
    for i in range(len(meas_data)):
        meas:Measurement = meas_data[i]
        # TODO: Fix assumption that measurement is synchronized with inputs
        if meas.stamp<=input_data[-1].stamp:
            x_k_idx = factored_stamp_list.index(meas.stamp)
            x_k:FactoredMLGState = factored_state_list[x_k_idx]
            proj_k = np.copy(x_k.projection)
            
            meas_factor = MeasurementFactor(mean=x_k.get_mean(),
                                            covariance=x_k.get_covariance(), stamp=meas.stamp,proj_matrix=proj_k,meas=meas, gh_degree=gh_deg, cubature_type=cubature_type)
            factored_state_list.append(meas_factor)

    return factored_state_list
                

            



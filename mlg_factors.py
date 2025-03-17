import numpy as np
import scipy.linalg
import navlie as nav
from typing import Callable, Optional, List, Tuple
from util.cubatures import gh_cubature, spherical_cubature
from navlie.lib.states import VectorState, State, MatrixLieGroupState, MatrixLieGroup, SE2State, CompositeState
from navlie.lib.models import PointRelativePosition
from navlie.types import ProcessModel, Measurement, Input, StateWithCovariance, MeasurementModel
from navlie.batch.problem import Problem
from navlie.utils import find_nearest_stamp_idx
from util.psd import force_PSD, force_sym
from abc import abstractmethod
from pymlg import SO2

class FactoredState:
    def __init__(self,mean:State, covariance:np.ndarray, proj_matrix:np.ndarray, stamp:float, related_factor: Optional['FactoredState'] = None,cubature_type='GH', gh_degree = 3):
        self.mean = mean.copy()
        self.mean.value = np.copy(mean.value).astype(np.float64)
        if isinstance(mean, MatrixLieGroupState):
            self.group = mean.group
        else:
            self.group = None
        self.covariance:np.ndarray = np.copy(covariance).astype(np.float64)
        self.sqrt_covariance:np.ndarray = np.linalg.cholesky(covariance)
        self.information:np.ndarray = force_PSD(scipy.linalg.inv(covariance))
        self.projection = np.copy(proj_matrix)
        self.stamp = stamp
        self.dof = self.mean.dof
        self.id = self.mean.state_id
        self.expect_scalar:np.ndarray = None
        self.expect_column:np.ndarray = None
        self.expect_matrix:np.ndarray = None
        self.sigma_pts:List[State] = []

        # Handle Related Factor
        self.related_factor = related_factor
        if related_factor is not None:
            self.total_dof = self.dof + related_factor.dof
            # TODO: Should this be a zero block at initiation?
            zero_block = np.zeros((self.related_factor.dof, self.dof))
            self.total_covariance = np.block([
                [related_factor.covariance, zero_block],
                [zero_block.T, self.covariance]])
            self.total_information = force_PSD(scipy.linalg.inv(self.total_covariance))
            self.total_projection = np.vstack((related_factor.projection, self.projection))
        else:
            self.total_dof = self.dof
            self.total_covariance = self.covariance
            self.total_information = self.information
            self.total_projection = self.projection
            
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
        self.sqrt_covariance = np.linalg.cholesky(self.total_covariance)
        # self.sqrt_covariance = scipy.linalg.sqrtm(self.total_covariance)
        vector_sigma_points = [self.sqrt_covariance @ sp_i.reshape((-1,1)) for sp_i in self._unit_sigma_pts]
        if self.related_factor is not None:
            self.sigma_pts = []
            for sp_vec in vector_sigma_points:
                sp_vec_prev = sp_vec[0:self.related_factor.dof]
                sp_vec_cur = sp_vec[self.dof:]
                sp_lie_prev = self.related_factor.mean.plus(sp_vec_prev)
                sp_lie_cur = self.mean.plus(sp_vec_cur)
                self.sigma_pts.append((sp_lie_prev, sp_lie_cur))
        else:
            self.sigma_pts = [self.mean.plus(sp_vec) for sp_vec in vector_sigma_points]

        return
    
    def phi_dx(self):
        return self.total_information @ self.expect_column

    def phi_dx_dx(self):
        a = self.total_information @ self.expect_matrix @ self.total_information
        b = self.total_information * self.expect_scalar
        return a - b
    
    def phi_dinfo(self):
        a = -0.5 * self.expect_matrix
        b = 0.5 * self.total_covariance * self.expect_scalar
        c = 0.5 * self.total_covariance
        # c = np.zeros_like(self.covariance)
        return a + b + c
    
    def compute_expectations(self):
        expect_mu_mu_phi = np.zeros_like(self.information)
        expect_mu_phi = np.zeros((self.dof, 1), dtype=np.float64)
        expect_phi = np.zeros((1,1))
        for i, w in enumerate(self.weights):
            phi_k_l = self.eval_phi(self.sigma_pts[i])
            expect_phi += w * phi_k_l
            
            diff = self.sigma_pts[i].minus(self.mean).reshape((-1,1))
            # diff = self.group.adjoint(self.mean.value) @ diff
            expect_mu_phi += w * (diff) * phi_k_l
            expect_mu_mu_phi += w * (diff) @ (diff.T)  * phi_k_l
        
        self.expect_scalar = expect_phi.copy()
        self.expect_column = expect_mu_phi.copy()
        self.expect_matrix = expect_mu_mu_phi.copy()

        return
    
    @abstractmethod
    def eval_phi(self, sigma_point:State) -> np.ndarray:
        pass

    @abstractmethod
    def update_factor(self, total_mean, total_information, total_covariance, delta_mean=None):
        pass

    def update_state(self, total_mean, total_information, total_covariance, delta_mean=None):
        # Project information, covariance
        self.information = self.projection @ np.copy(total_information) @ self.projection.T
        self.covariance = self.projection @ np.copy(total_covariance) @ self.projection.T

        # Project the mean, correct information covariance
        mean_vector = self.projection @ np.copy(total_mean)
        if self.group is not None:
            if delta_mean is None:
                self.mean.value = self.group.Exp(mean_vector)
            else:
                self.mean:MatrixLieGroupState
                delta_mean_vector = self.projection @ np.copy(delta_mean)
                self.mean.plus(delta_mean_vector)
                if self.mean.direction == 'left':
                    jac = self.group.left_jacobian(delta_mean)
                    jac_inv = self.group.left_jacobian_inv(delta_mean)
                elif self.mean.direction == 'right':
                    jac = self.group.right_jacobian(delta_mean)
                    jac_inv = self.group.right_jacobian_inv(delta_mean)
                self.information = jac.T @ self.information @ jac
                self.covariance = jac_inv @ self.covariance @ jac_inv.T

        else:
            self.mean.value = np.copy(mean_vector)
        
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
            # C, r = self.group.to_components(self.mean.value)
            # theta = SO2.Log(C)
            # return np.vstack((theta, r.reshape((-1,1))))
            return np.copy(self.group.Log(self.mean.value).reshape((self.dof, 1)))
        else:
            return np.copy(self.mean.value.reshape((self.dof, 1)))
        

class ProcessFactor(FactoredState):
    def __init__(self, mean:State, covariance:np.ndarray, proj_matrix:np.ndarray, related_factor:FactoredState, process_model:ProcessModel, u:Input, stamp, cubature_type='GH', gh_degree=3):
        super().__init__(mean, covariance, proj_matrix, stamp, related_factor=related_factor, cubature_type=cubature_type, gh_degree=gh_degree)
        self.sigma_pts:List[Tuple[State]]
        # Set process specifics
        self.process_model = process_model
        self.u = u
        self.dt = self.stamp - self.u.stamp
        self._Q = self.process_model.covariance(self.mean, self.u, self.dt)
        self._Q_inv = force_PSD(scipy.linalg.inv(self._Q))
        
        # Compute initial expectations using the Sigma Points
        self.compute_expectations()
        return
    
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
    
    def update_factor(self, total_mean, total_information, total_covariance, delta_mean=None):
        # Update total covar/info across both factors
        self.total_information = self.total_projection @ np.copy(total_information) @ self.total_projection.T
        self.total_covariance = self.total_projection @ np.copy(total_covariance) @ self.total_projection.T
        # Fix factor covar/info during retraction
        if (delta_mean is not None) and (self.group is not None):
            self.mean:MatrixLieGroupState
            delta_mean:np.ndarray
            mean_prev = delta_mean[0:self.related_factor.dof]
            mean_cur = delta_mean[self.related_factor.dof:]
            if self.mean.direction == 'left':
                jac_prev = self.group.left_jacobian(mean_prev)
                jac_cur = self.group.left_jacobian(mean_cur)  
                jac_prev_inv = self.group.left_jacobian_inv(mean_prev)
                jac_cur_inv = self.group.left_jacobian_inv(mean_cur)
            elif self.mean.direction == 'right':
                jac_prev = self.group.right_jacobian(mean_prev)
                jac_cur = self.group.right_jacobian(mean_cur)
                jac_prev_inv = self.group.right_jacobian_inv(mean_prev)
                jac_cur_inv = self.group.right_jacobian_inv(mean_cur)
                
            jac = scipy.linalg.block_diag(jac_prev, jac_cur)
            jac_inv = scipy.linalg.block_diag(jac_prev_inv, jac_cur_inv)
            self.total_information = jac.T @ self.total_information @ jac
            self.total_covariance = jac_inv @ self.total_covariance @ jac_inv.T

        
        # Update state
        self.update_state(total_mean, total_information, total_covariance, delta_mean=delta_mean)

        # Recompute sigma points around new mean / covariance
        self.generate_new_sigma_pts()
        # Recompute expectations using new sigma points
        self.compute_expectations()

    def get_cross_information(self):
        return np.copy(self.total_information[0:self.related_factor.dof, self.dof:])
    
    def get_cross_covariance(self):
        return np.copy(self.total_covariance[0:self.related_factor.dof, self.dof:])
    
    def set_cross_information(self, cross_info:np.ndarray):
        """
        Sets the top right cross-information term. 
        """
        if cross_info.shape != (self.related_factor.dof, self.dof):
            raise ValueError("Incorrectly sized cross information.")
        
        self.total_information[0:self.related_factor.dof, self.dof:] = np.copy(cross_info)
        self.total_information[self.dof:, 0:self.related_factor.dof] = np.copy(cross_info.T)
        return
      
    
class PriorFactor(FactoredState):
    def __init__(self, mean:State, covariance:np.ndarray, proj_matrix:np.ndarray, prior:StateWithCovariance, stamp, cubature_type='GH', gh_degree=3):
        super().__init__(mean, covariance, proj_matrix, stamp, cubature_type=cubature_type, gh_degree=gh_degree)  
        # Setup prior terms
        self.x0_check = prior.state.copy()
        self.x0_check.value = np.copy(prior.state.value)
        self.P0_check = np.copy(prior.covariance)
        self.P0_check_inv = force_PSD(scipy.linalg.inv(self.P0_check))
        # Compute relevant expectations
        self.compute_expectations()

    def eval_phi(self, sigma_point):
        prior_diff = sigma_point.minus(self.x0_check).reshape((self.dof, 1))
        phi_prior = 0.5 * prior_diff.T @ self.P0_check_inv @ prior_diff
        return phi_prior
    
    def update_factor(self, total_mean, total_information, total_covariance, delta_mean=None):
        # Doesn't have any dependence on another factor, so can just update the individual state
        return super().update_state(total_mean, total_information, total_covariance, delta_mean)
    
class LandmarkPriorFactor(PriorFactor):
    def __init__(self, mean:State, covariance:np.ndarray, proj_matrix:np.ndarray, prior:StateWithCovariance, stamp, cubature_type='GH', gh_degree=3):
        super().__init__(mean, covariance, proj_matrix, prior, stamp, cubature_type, gh_degree)

    
    
class MeasurementFactor(FactoredState):
    def __init__(self, mean, covariance, proj_matrix, meas:Measurement, stamp, related_factor: Optional['FactoredState']=None,cubature_type='GH', gh_degree=3):
        super().__init__(mean, covariance, proj_matrix, stamp, related_factor, cubature_type, gh_degree) 
        self.y_k = np.copy(meas.value)
        self.meas_model = meas.model
        self.R_k = np.atleast_2d(meas.model.covariance(self.mean))
        self.R_k_inv = force_PSD(scipy.linalg.inv(self.R_k))
        self.compute_expectations()

    def eval_phi(self, sigma_point):
        # Calculate measurement model propagated difference
        meas_diff = (self.y_k - self.meas_model.evaluate(sigma_point)).reshape((-1,1))
        phi_meas = 0.5 * meas_diff.T @ self.R_k_inv @ meas_diff
        return phi_meas
    
    def update_factor(self, total_mean, total_information, total_covariance, delta_mean=None):
        # Doesn't have any dependence on another factor, so can just update the individual state
        return super().update_state(total_mean, total_information, total_covariance, delta_mean)
    
class MeasurementSLAMFactor(MeasurementFactor):
    def __init__(self, mean, covariance, proj_matrix, meas:Measurement, stamp, related_factor:FactoredState,cubature_type='GH', gh_degree=3):
        super().__init__(mean, covariance, proj_matrix, meas, stamp, related_factor, cubature_type, gh_degree)
        self.sigma_pts:List[Tuple[State]]
        
    # TODO: Turn this into composite state
    def eval_phi(self, cur_sp:State, landmark_sp:State):
        comp_state = CompositeState([cur_sp, landmark_sp], stamp=self.stamp)
        meas_diff = self.y_k - self.meas_model.evaluate(comp_state)
        phi_meas = 0.5 * meas_diff.T @ self.R_k_inv @ meas_diff
        return phi_meas
    
    def compute_expectations(self):
        total_dim = self.total_dof
        expect_mu_mu_phi = np.zeros((total_dim, total_dim))
        expect_mu_phi = np.zeros((total_dim, 1))
        expect_phi = np.zeros((1,1))
        for i, w in enumerate(self.weights):
            landmark_sp = self.sigma_pts[i][0]
            cur_sp = self.sigma_pts[i][1]
            phi_k_l = self.eval_phi(cur_sp=cur_sp, landmark_sp=landmark_sp)
            expect_phi += w * phi_k_l
            landmark_diff = landmark_sp.minus(self.related_factor.mean).reshape((-1,1))
            cur_diff = cur_sp.minus(self.mean).reshape((-1,1))
            diff = np.vstack((landmark_diff, cur_diff))
            expect_mu_phi += w * phi_k_l * diff
            expect_mu_mu_phi += w * phi_k_l * (diff @ diff.T)

        self.expect_scalar = np.copy(expect_phi)
        self.expect_column = np.copy(expect_mu_phi)
        self.expect_matrix = np.copy(expect_mu_mu_phi)
        return 
    
    def update_factor(self, total_mean, total_information, total_covariance, delta_mean=None):
        # Update total covar/info across both factors
        self.total_information = self.total_projection @ np.copy(total_information) @ self.total_projection.T
        self.total_covariance = self.total_projection @ np.copy(total_covariance) @ self.total_projection.T
        # Fix factor covar/info during retraction
        if (delta_mean is not None) and (self.group is not None):
            self.mean:MatrixLieGroupState
            delta_mean:np.ndarray
            mean_prev = delta_mean[0:self.related_factor.dof]
            mean_cur = delta_mean[self.related_factor.dof:]
            if self.mean.direction == 'left':
                jac_prev = np.eye(self.related_factor.dof)
                jac_cur = self.group.left_jacobian(mean_cur)  
                jac_prev_inv = np.eye(self.related_factor.dof)
                jac_cur_inv = self.group.left_jacobian_inv(mean_cur)
            elif self.mean.direction == 'right':
                jac_prev = np.eye(self.related_factor.dof)
                jac_cur = self.group.right_jacobian(mean_cur)
                jac_prev_inv = np.eye(self.related_factor.dof)
                jac_cur_inv = self.group.right_jacobian_inv(mean_cur)
                
            jac = scipy.linalg.block_diag(jac_prev, jac_cur)
            jac_inv = scipy.linalg.block_diag(jac_prev_inv, jac_cur_inv)
            self.total_information = jac.T @ self.total_information @ jac
            self.total_covariance = jac_inv @ self.total_covariance @ jac_inv.T
        
        # Update state
        self.update_state(total_mean, total_information, total_covariance, delta_mean=delta_mean)

        # Recompute sigma points around new mean / covariance
        self.generate_new_sigma_pts()
        # Recompute expectations using new sigma points
        self.compute_expectations()
    
    def get_cross_information(self):
        return np.copy(self.total_information[0:self.related_factor.dof, self.dof:])
    
    def get_cross_covariance(self):
        return np.copy(self.total_covariance[0:self.related_factor.dof, self.dof:])
    
    def set_cross_information(self, cross_info:np.ndarray):
        """
        Sets the top right cross-information term. 
        """
        if cross_info.shape != (self.related_factor.dof, self.dof):
            raise ValueError("Incorrectly sized cross information.")
        
        self.total_information[0:self.related_factor.dof, self.dof:] = np.copy(cross_info)
        self.total_information[self.dof:, 0:self.related_factor.dof] = np.copy(cross_info.T)
        return


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
        x_k.state_id = 'x' + str(i+1)
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
            fac_k_idx = factored_stamp_list.index(meas.stamp)
            fac_k:FactoredState = factored_state_list[fac_k_idx]
            proj_k = np.copy(fac_k.projection)
            
            meas_factor = MeasurementFactor(mean=fac_k.get_mean(),
                                            covariance=fac_k.get_covariance(), stamp=meas.stamp,proj_matrix=proj_k,meas=meas, gh_degree=gh_deg, cubature_type=cubature_type)
            factored_state_list.append(meas_factor)

    return factored_state_list

def construct_slam_factor_list(x0:StateWithCovariance, input_data:List[Input], meas_data:List[Measurement], landmark_data:List[StateWithCovariance],proc_model:ProcessModel, meas_model:MeasurementModel, cubature_type ='GH', gh_deg = 3):
    factored_state_list = []
    factored_landmark_dict = {}
    factored_stamp_list = []
    state_dof = x0.state.dof
    landmark_dof = landmark_data[0].state.dof
    state_factors_dof = len(input_data)*state_dof
    landmark_factors_dof = len(landmark_data) * landmark_dof
    total_factors_dof = state_factors_dof + landmark_factors_dof

    # Define prior
    proj_0 = np.zeros((state_dof, total_factors_dof))
    proj_0[:, :state_dof] = np.eye(state_dof)
    prior_factor = PriorFactor(mean=x0.state.copy(), covariance=x0.covariance, proj_matrix=proj_0, prior=x0.copy(), stamp=x0.state.stamp, gh_degree=gh_deg, cubature_type=cubature_type)
    factored_state_list.append(prior_factor)
    factored_stamp_list.append(x0.state.stamp)

    # Define landmark priors
    landmark_idx = 0
    for l in landmark_data:
        proj_0_landmark = np.zeros((landmark_dof, total_factors_dof))
        idx = state_factors_dof+landmark_idx
        proj_0_landmark[:, idx:idx+landmark_dof] = np.eye(landmark_dof)
        landmark_factor = LandmarkPriorFactor(mean=l.state.copy(), covariance=l.covariance, proj_matrix=proj_0_landmark, prior=l.copy(), stamp=None, cubature_type=cubature_type, gh_degree=gh_deg)
        factored_landmark_dict[l.state.state_id] = landmark_factor

    # Define process prior
    proj_idx = state_dof
    x_k_1 = x0.state.copy()
    P_k_1 = force_sym(x0.covariance)
    for i in range(len(input_data)-1):
        u_k_1:Input = input_data[i]
        u_k:Input = input_data[i+1]
        dt_u = u_k.stamp - u_k_1.stamp
        x_k:State = proc_model.evaluate(x_k_1, u=u_k_1, dt=dt_u)
        x_k.state_id = 'x' + str(i+1)
        x_k.stamp = u_k.stamp
        A_k = proc_model.jacobian(x_k_1, u=u_k_1, dt=dt_u)
        Q_k = proc_model.covariance(x=x_k_1, u=u_k_1, dt=dt_u)
        P_k = A_k @ P_k_1 @ A_k.T + Q_k
        P_k = force_sym(P_k)
        proj_k = np.zeros((state_dof, total_factors_dof))
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
            
            fac_k_idx = factored_stamp_list.index(meas.stamp)
            fac_k:FactoredState = factored_state_list[fac_k_idx]
            proj_k = np.copy(fac_k.projection)
            landmark_id = meas.model._landmark_state_id
            meas.model = meas_model(fac_k.mean.state_id, landmark_id, meas.model.covariance(None))
            landmark_factor = factored_landmark_dict.get(landmark_id)
            meas_factor = MeasurementSLAMFactor(mean=fac_k.get_mean(),
                                            covariance=fac_k.get_covariance(), proj_matrix=proj_k, meas=meas, stamp=meas.stamp, related_factor=landmark_factor, cubature_type=cubature_type, gh_degree=gh_deg)
            factored_state_list.append(meas_factor)
    
    for landmark in landmark_data:
        factored_state_list.append(factored_landmark_dict[landmark.state.state_id])
    

    return factored_state_list

def factor_list_from_map(opt_variables, problem:Problem, input_data:List[nav.types.Input], meas_data:List[Measurement], proc_model:ProcessModel, cubature_type ='GH', gh_deg = 3) -> List[FactoredState]:      
    #TODO: Fix this given landmark states
    factored_state_list = []
    factored_stamp_list = []

    # Prior Factors
    x0_state:State = opt_variables['x0']
    state_dof = x0_state.dof
    P0 = problem.get_covariance_block(x0_state.state_id, x0_state.state_id)

    proj_0 = np.zeros((state_dof, len(input_data)*state_dof))
    proj_0[:,:state_dof] = np.eye(state_dof)

    prior_factor = PriorFactor(mean=x0_state.copy(), covariance=P0, proj_matrix=proj_0, prior=StateWithCovariance(state=x0_state.copy(), covariance=P0), stamp=x0_state.stamp, cubature_type=cubature_type, gh_degree=gh_deg)
    factored_state_list.append(prior_factor)
    factored_stamp_list.append(x0_state.stamp)

    # x_k_1 = x0_state.copy()
    # P_k_1 = np.copy(P0)
    proj_idx = state_dof
    # Add Process Factors
    for i in range(len(input_data)-1):
        u_k_1:Input = input_data[i]
        u_k:Input = input_data[i+1]
        dt_u = u_k.stamp - u_k_1.stamp
        
        x_k:State = opt_variables['x'+str(i+1)].copy()
        P_k = problem.get_covariance_block(x_k.state_id, x_k.state_id)
        #TODO: Change sizing to account for landmarks
        proj_k = np.zeros((state_dof, len(input_data)*state_dof))
        proj_k[:, proj_idx:proj_idx+state_dof] = np.eye(state_dof)
        process_fac_k = ProcessFactor(mean=x_k.copy(), covariance=P_k, proj_matrix=proj_k, related_factor=factored_state_list[-1], process_model=proc_model, u=u_k_1, stamp=x_k.stamp, cubature_type=cubature_type, gh_degree=gh_deg)
        # TODO: Set cross information values
        # process_fac_k.set_cross_information(cross_info=problem.get_covariance_block())
        factored_state_list.append(process_fac_k)
        factored_stamp_list.append(x_k.stamp)
        proj_idx += state_dof
    
    # Add Measurement Factors
    for i in range(len(meas_data)):
        meas:Measurement = meas_data[i]
        if meas.stamp<=input_data[-1].stamp:
            x_k_idx = find_nearest_stamp_idx(factored_stamp_list, meas.stamp)
            x_k:FactoredState = factored_state_list[x_k_idx]
            proj_k = np.copy(x_k.projection)
            # TODO: Add landmark factors here eventually
            meas_factor = MeasurementFactor(mean=x_k.get_mean(), covariance=x_k.get_covariance(), stamp=meas.stamp, proj_matrix=proj_k, meas=meas, cubature_type=cubature_type, gh_degree=gh_deg)
            factored_state_list.append(meas_factor)
    
    return factored_state_list




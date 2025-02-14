import numpy as np
import scipy.linalg
import navlie as nav
from typing import Callable, Optional, List
from util.cubatures import gh_cubature, spherical_cubature
from navlie.lib.states import VectorState, State, MatrixLieGroupState, MatrixLieGroup
from navlie.types import ProcessModel, Measurement, Input, StateWithCovariance
from util.psd import force_PSD, force_sym
from abc import abstractmethod

class FactoredMLGState:
    def __init__(self, mean:MatrixLieGroupState, covariance:np.ndarray, proj_matrix:np.ndarray, stamp:float, cubature_type= 'GH', gh_degree = 3):
        self.mean = mean.copy()
        self.mean.value = np.copy(mean.value)
        self.group = mean.group
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
        self._sigma_pts:List[MatrixLieGroupState]

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
        

        return
    
    def generate_new_sigma_pts(self):
        self.sqrt_covariance = np.linalg.cholesky(self.covariance)
        # Now Sigma Points will be on Lie Group
        # So will also be list of MatrixLieGroup
        self._sigma_pts = [self.mean.plus(self.sqrt_covariance @ sp_i.reshape((-1,1)))
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
        for i, w in enumerate(self._weights):
            phi_k_l = self.eval_phi(self._sigma_pts[i])
            expect_phi += w * phi_k_l
            expect_mu_phi += w * (self._sigma_pts[i].minus(self.mean)) * phi_k_l
            expect_mu_mu_phi += w * (self._sigma_pts[i].minus(self.mean)) @ (self._sigma_pts[i].minus(self.mean)).T  * phi_k_l
        
        self.expect_scalar = expect_phi.copy()
        self.expect_column = expect_mu_phi.copy()
        self.expect_matrix = expect_mu_mu_phi.copy()

        return
    
    @abstractmethod
    def eval_phi(self, sigma_point:MatrixLieGroupState) -> np.ndarray:
        pass

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

    # Get functions
    def get_mean(self):
        return self.mean.copy()
    
    def get_information(self):
        return self.information.copy()
    
    def get_covariance(self):
        return self.covariance.copy()
        


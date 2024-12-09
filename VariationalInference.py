# %%
import numpy as np
import navlie as nav
from typing import Callable, Optional, List

from gh_quad_ex import gh_cubature_nav

from math import factorial
import itertools
from numpy.polynomial.hermite_e import hermeroots
from scipy.stats.distributions import chi2
from scipy.special import eval_hermitenorm


class GVI:
    def __init__(self, num_states:int, state_dim:int, gh_degree:int, phi_function: Callable) -> None:
        self.state_dim = state_dim
        self.num_states = num_states
        self.gvi_dimension = state_dim * num_states

        self._phi_func = phi_function

        # Initialize placeholders
        self.mu = np.zeros(self.gvi_dimension, dtype=np.float64).reshape((-1,1))
        self.covariance = np.eye(self.gvi_dimension, dtype=np.float64)
        self.information = np.linalg.inv(self.covariance)
        self._sqrt_cov = np.linalg.cholesky(self.covariance)
        
        # Gauss-Hermite Integration Setup
        self._gh_dim = self.state_dim ** self.num_states
        self._gh_degree = gh_degree

    def set_initial(self, mu_0:np.ndarray, covar_0:np.ndarray):
        self.mu = mu_0.reshape((-1,1)).astype(np.float64)
        self.covariance = covar_0.reshape((self.gvi_dimension, self.gvi_dimension)).astype(np.float64)
        self.information = np.linalg.inv(self.covariance)
        self._sqrt_cov = np.linalg.cholesky(self.covariance)
        return
    

    def run(self):
        n_iters = 0
        # print(self.mu.shape)
        while(True):
            # Initial Cost
            init_cost = self.eval_cost_func()
            
            
            # Calc Gradients
            information_new = self.comp_phi_ddx()
            information_new = self._force_psd(information_new)
            covariance_new = np.linalg.inv(information_new)
            covariance_new = self._force_psd(covariance_new)
            sqrt_cov_new = np.linalg.cholesky(covariance_new)
            

            # Compute Deltas
            info_close = np.allclose(information_new, self.information)
            delta_info = information_new - self.information
            delta_covar = covariance_new - self.covariance
            delta_mu = - covariance_new @ self.comp_phi_dx()
            self.mu += delta_mu
            
            if abs(np.linalg.norm(delta_mu))<1e-6 and info_close:
                print(n_iters)
                break
            if n_iters > 1000:
                print(n_iters)
                print(delta_mu)
                print(delta_info)
                break
            # Backtracking Line Search
            # New Cost
            new_cost = np.copy(init_cost)
            
            # alpha = 1
            
            # while new_cost >= init_cost:
            #     # Update 
            #     self.mu += alpha*delta_mu
            #     self.information+= alpha*delta_info
            #     self.covariance = alpha*delta_covar
            #     self.covariance = self._force_psd(self.covariance)
            #     self._sqrt_cov = sqrt_cov_new
            #     # Calc New Cost
            #     new_cost = self.eval_cost_func()
            #     alpha *= 0.9
            #     print(alpha)
            
            n_iters += 1
            print(self.mu)
            print(self.covariance)
            print(n_iters, '\n ----------------- \n')
        return self.mu, self.covariance

    def eval_cost_func(self):
        a = - self.comp_phi()
        b = 0.5 * np.log(np.linalg.det(self.information))
        return a + b

    def _gh_integrate(self, expect_func:Callable):
        unit_sigma_pts, weights = gh_cubature_nav(p=self._gh_degree, dof=self._gh_dim)
        self._sqrt_cov = np.linalg.cholesky(self.covariance)
        sigma_points = [self.mu.reshape((-1,1)) + self._sqrt_cov @ sp_i.reshape((-1,1)) 
                for sp_i in unit_sigma_pts.T]
        expect = expect_func(sigma_points, weights)
        return expect
    
    def comp_phi(self):
        return self._gh_integrate(self._expect_phi)
    
    def comp_phi_dx(self):
        return self.information @ self._gh_integrate(self._expect_xmmu_phi)
    
    def comp_phi_ddx(self):
        a = self.information @ self._gh_integrate(self._expect_xmmu_2_phi) @ self.information
        b = self.information @ self._gh_integrate(self._expect_phi)
        return a - b
    
    def _expect_phi(self, x_k:List[np.ndarray], weights:List[np.ndarray]):
        # E_{q_k}[\phi_k (x)k]
        # TODO: Check dimensions of everything
        expect = np.zeros_like(x_k[0])
        for i, w in enumerate(weights):
            expect += self._phi_func(x_k[i]) * w
        return expect
    
    def _expect_xmmu_phi(self, x_k:np.ndarray, weights:List[np.ndarray]):
        # E_{q_k}[(x_k - \mu_k )\phi_k (x)k]
        # TODO: Check dimensions of what self._phi_func returns
        expect = np.zeros_like(x_k[0])
        for i, w in enumerate(weights):
            expect += w*(x_k[i] - self.mu.reshape((-1,1)))*self._phi_func(x_k[i])
        return expect
    
    def _expect_xmmu_2_phi(self, x_k:np.ndarray, weights:List[np.ndarray]):
        expect = np.zeros_like(self.information)
        # print(np.shape(expect))
        # print(np.shape(x_k[0] - self.mu))
        # TODO: Checl dimensions of what self._phi_func
        for i, w in enumerate(weights):
            expect += self._force_psd(w*((x_k[i] - self.mu.reshape((-1,1))) @ (x_k[i] - self.mu.reshape((-1,1))).T ) * self._phi_func(x_k[i]))
        return expect
    
    def _force_psd(self, matrix):
        eigvals, eigvecs = np.linalg.eig(matrix.T)
        # print(eigvals)
        eigvals[eigvals <= 0] = 1e-10  # Adjust negative eigenvalues
        psd = eigvecs @ np.diag(eigvals) @ eigvecs.T
        sym = 0.5 * (psd + psd.T)
        return sym
# %%
import numpy as np
import navlie as nav
from typing import Callable, Optional, List

from gh_quad_ex import gh_cubature_nav, gh_cubature

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
        # Only need at x_k dimension
        self._gh_dim = self.state_dim
        self._gh_degree = gh_degree
        self._unit_sigma_pts, self._weights = gh_cubature(order_p=self._gh_degree, state_dof=self._gh_dim)

    def set_initial(self, mu_0:np.ndarray, covar_0:np.ndarray):
        self.mu = mu_0.reshape((-1,1)).astype(np.float64)
        self.covariance = covar_0.reshape((self.gvi_dimension, self.gvi_dimension)).astype(np.float64)
        self.information = np.linalg.inv(self.covariance)
        self._sqrt_cov = np.linalg.cholesky(self.covariance)
        return
    

    def run(self, debug=False):
        n_iters = 1
        # print(self.mu.shape)
        while(True):
            # Update section

            # Get new sigma pts centered on current mean
            self._generate_new_sigma_pts()
            # Compute new information and covariance
            new_information = self.phi_dx_dx()
            new_covariance = np.linalg.inv(new_information)
            delta_mu = -1 * new_covariance @ self.phi_dx()
            new_mu = self.mu + delta_mu
            
            # Calculate breaking condition
            size_mu = np.abs(np.linalg.norm(delta_mu))
            delta_info = new_covariance @ self.information
            size_info = np.linalg.norm(new_information - self.information)
            if size_mu < 1e-6 and size_info < 1e-6:
                if debug:
                    print("--------------------------------")
                    print(f"|  Converged in {n_iters} iterations!  |")
                    print("--------------------------------")

                break
            if n_iters > 10000:
                if debug:
                    print(f"Reached max iterations")
                break
            
            
            self.information = self._force_psd(new_information)
            self.covariance = self._force_psd(new_covariance)
            self.mu = new_mu
            if debug:
                print( "########################################")
                print("Iteration: ", n_iters)
                print( "----------------------------------------")
                print("Delta mu: ", delta_mu.T)
                print("Delta Info: ", delta_info)
                print("Info_{i+1}: ", self.information)
                print( "########################################\n")
            n_iters += 1
                

        return self.mu, self.covariance

    def eval_cost_func(self):
        # TODO: Check this negative sign !!
        a = - self._expect_phi()
        b = 0.5 * np.log(np.linalg.det(self.information))
        return a + b

    def _generate_new_sigma_pts(self):
        self._sqrt_cov = np.linalg.cholesky(self.covariance)
        self._sigma_pts = [self.mu.reshape((-1,1)) + self._sqrt_cov @ sp_i.reshape((-1,1)) 
                for sp_i in self._unit_sigma_pts]
        return
    
    def _expect_phi(self):
        # Scalar
        expect = 0
        for i, w in enumerate(self._weights):
            expect += w * self._phi_func(self._sigma_pts[i]) 
        return expect
    
    def _expect_mu_phi(self):
        # TODO: Change this for mu_k
        expect = np.zeros_like(self.mu)
        for i, w in enumerate(self._weights):
            expect += w * (self._sigma_pts[i] - self.mu) * self._phi_func(self._sigma_pts[i])
        return expect
    
    def _expect_mu_mu_phi(self):
        # TODO: Change this for information_kk
        expect = np.zeros_like(self.information)
        for i, w in enumerate(self._weights):
            expect += w * (self._sigma_pts[i] - self.mu) @ (self._sigma_pts[i] - self.mu).T  * self._phi_func(self._sigma_pts[i])
        return expect
    
    def phi_dx(self):
        return self.information @ self._expect_mu_phi()
    
    def phi_dx_dx(self):
        # TODO: Change this for information_k
        a = self.information @ self._expect_mu_mu_phi() @ self.information
        b = self.information * self._expect_phi()
        if np.linalg.norm(np.abs(a-b)) < 1e-8:
            print("Information: ", self.information)
            print("Covariance: ",self.covariance)
            print("Mean: ",self.mu)
            print("Sigma Points/Weights: ",self._sigma_pts, self._weights)
            print("Expect (x - mu)(x-mu).T phi(x)", self._expect_mu_mu_phi())
            print("Expect phi(x)", self._expect_phi())
            print("a", a)
            print("b", b)
            raise ValueError("Information Update is Singular")
        return a - b

    
    def _force_psd(self, matrix):
        eigvals, eigvecs = np.linalg.eig(matrix.T)
        # print(eigvals)
        eigvals[eigvals <= 0] = 1e-10  # Adjust negative eigenvalues
        psd = eigvecs @ np.diag(eigvals) @ eigvecs.T
        sym = 0.5 * (psd + psd.T)
        return sym


def cost_function_1D(vec_x):
    x = vec_x[0,0]
    mu_x = 20
    f = 400
    b = 0.1
    sig_y_sq = 0.09
    sig_x_sq = 9

    # y should be sampled. For a single trial, just give it a value.
    y = f * b / 26 - 0.6

    return ((x - mu_x)**2 / (2 * sig_x_sq) + (y - f * b / x)**2 / (2 * sig_y_sq))

def cost_function_2D(x:np.ndarray):
    height=7 
    distance=3
    range_measure = lambda x: np.sqrt(np.square(x[0] + distance) + np.square(height))
    expect_x = np.array([[2],[1]])
    expect_y = range_measure(expect_x)
    Q = np.diag([0.2, 0.2])
    R = np.eye(1)*0.1
    # y = range_measure(expect_x) + np.sqrt(var_y)*np.random.randn(var_y.shape[0], var_y.shape[1])
    y = np.array([[8.40381514]])
    phi_x = 0.5 * (x - expect_x).T @ np.linalg.inv(Q) @ (x - expect_x)
    phi_y = 0.5 * (y - expect_y).T @ np.linalg.inv(R) @ (y - expect_y)
    return phi_x + phi_y
    
# # %%
gvi = GVI(num_states=1, state_dim=1, gh_degree=3, phi_function=cost_function_1D)
covar_0 = np.diag([9.1])
mean_0 = np.array([[20]])
gvi.set_initial(mean_0, covar_0=covar_0)
mean_pred, covar_pred = gvi.run(debug=True)

# %%
gvi = GVI(num_states=1, state_dim=2, gh_degree=3, phi_function=cost_function_2D)
covar_0 = np.diag([2, 1.5])
mean_0 = np.array([[2.2],[1.2]])
gvi.set_initial(mean_0, covar_0=covar_0)
mean_pred, covar_pred = gvi.run(debug=True)



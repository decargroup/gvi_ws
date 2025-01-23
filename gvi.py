# %%
import numpy as np
import scipy.linalg
import navlie as nav
from typing import Callable, Optional, List
from util.cubatures import gh_cubature, spherical_cubature
from factors import FactoredState, PriorFactor, ProcessFactor, MeasurementFactor
from models import Simulator, NonLinearLaserRangeFinder, LaserRangeFinder, DoubleIntegrator
from navlie.datagen import DataGenerator
from navlie.lib.states import VectorState, VectorInput
from navlie.lib.models import RangePointToAnchor
from navlie.types import ProcessModel, MeasurementModel, Measurement, Input, StateWithCovariance, State
from util.psd import force_PSD
import matplotlib.pyplot as plt
from scipy.linalg import block_diag
from abc import abstractmethod

class GVI:
    def __init__(self, factored_states:List[FactoredState], total_dim:int, debug=False):
        self.factored_states = factored_states
        self.total_dim = total_dim
        self.debug = debug
        self.factor_dof = 0
        # Initialize mean and information
        self.mean = np.zeros((total_dim, 1))
        self.information = np.zeros((total_dim, total_dim))
        self.covariance = np.zeros((total_dim, total_dim))
        self.cur_cost = np.inf

        # Build full mean and information
        k = 0
        phi = np.zeros((1,1))
        for x_k in factored_states:
            phi += x_k.expect_scalar
            if isinstance(x_k, PriorFactor):
                self.factor_dof = x_k.dof
            
            if isinstance(x_k, ProcessFactor):
                if k == 0:
                    dof = x_k.dof
                    state_dof = x_k.state_dof
                    self.mean[k:k+dof] = x_k.mean
                    self.information[k:k+dof, k:k+dof] = x_k.information
                    k+=dof
                else:
                    state_dof = x_k.state_dof
                    self.mean[k:k+state_dof] = x_k.get_mean()
                    self.information[k:k+state_dof, k:k+state_dof] = x_k.get_information()
                    # Cross Information Terms
                    self.information[k-state_dof:k, k:k+state_dof] = x_k.information[0:state_dof,state_dof:]
                    self.information[k:k+state_dof, k-state_dof:k] = x_k.information[state_dof:, 0:state_dof]
                    k+=state_dof
        self.information = force_PSD(self.information)
        L, D, _ = scipy.linalg.ldl(self.information, lower=True)
        self.covariance = self.compute_covariance(L, D)
        self.cur_cost = phi.copy() + np.linalg.slogdet(self.information)[1]

                
    def solve(self):
        n_iters = 0
        if self.debug:
            print("Starting Conditions: ")
            print("mu_{0}: \n", self.mean.T)
            print("Info_{0}: \n", self.information)
        while(True):
            # TODO: Implement backtracking
            # Update info
            phi_dx = np.zeros_like(self.mean)
            phi = np.zeros((1,1))
            new_information = np.zeros_like(self.information)
            # functional_dinfo = np.zeros_like(self.information)
            expect_matrix = np.zeros_like(self.information)
            for x_k in self.factored_states:
                # factored_state_k.generate_new_sigma_pts()
                proj_k = x_k.projection
                phi += x_k.expect_scalar
                phi_dx += proj_k.T @ x_k.phi_dx()
                new_information += proj_k.T @ force_PSD(x_k.phi_dx_dx()) @ proj_k
                # functional_dinfo += proj_k.T @ force_PSD(x_k.phi_dinfo()) @ proj_k
                expect_matrix += proj_k.T @ x_k.expect_matrix @ proj_k
            
            # May remove force_sparsity
            new_information = self._force_sparsity(new_information, deg=self.factor_dof)
            L, D, _ = scipy.linalg.ldl(new_information, lower=True)
            self.new_cost = phi + (0.5 * np.linalg.slogdet(D)[1])
            new_covariance = self.compute_covariance(L, D)
            # This has been a bit stabler
            delta_mu = scipy.linalg.solve(new_information, -1*phi_dx)

            # solve triangular unstable for some values
            # v = scipy.linalg.solve_triangular(L, -phi_dx, lower=True) 
            # delta_mu = scipy.linalg.solve_triangular(L.T, v, lower=False) 
            
            functional_dinfo = (-0.5*expect_matrix) + (0.5*self.covariance*phi) + (0.5*self.covariance)
            delta_info = force_PSD(-2 * self.information @ functional_dinfo @ self.information)
            
            # delta_info = self._force_sparsity(delta_info, deg=self.factor_dof)
            # delta_info = force_PSD(new_information - self.information)
        
            # Calculate breaking condition
            size_mu = np.abs(np.linalg.norm(delta_mu))
            size_info = np.linalg.trace(delta_info)

            if size_mu < 1e-6 and size_info < 1e-6:
                print("--------------------------------")
                print(f"|  Converged in {n_iters} iterations!  |")
                print("--------------------------------")

                break
            if n_iters > 10:
                print(f"Reached max iterations")
                print("|Info|: ", size_info)
                print("|mu|: ", size_mu)
                break
            
            # if self.new_cost > self.cur_cost:
            #     print("--------------------------------")
            #     print(f"|  Cost not reduced from old {self.cur_cost} to {self.new_cost}  |")
            #     print(f"|  After {n_iters} iterations!  |")
            #     print("--------------------------------")
            #     break

            # Backtracking loop
            alpha = 0.95
            backtrack_count = 0
            proposed_info = new_information.copy()
            proposed_covar = new_covariance.copy()
            proposed_mean = (self.mean + delta_mu).copy()
            while(self.new_cost >= self.cur_cost):
                backtrack_count += 1
                temp_phi = 0
                proposed_info = force_PSD(self.information + (alpha*delta_info))
                size_info = np.linalg.trace(alpha*delta_info)
                proposed_info = self._force_sparsity(proposed_info, self.factor_dof)
                
                L, D, _ = scipy.linalg.ldl(proposed_info, lower=True)
                proposed_covar = self.compute_covariance(L,D)
                proposed_mean = self.mean + (alpha*delta_mu)
                
                for x_k in self.factored_states:
                    x_k.update_factor(total_mean=proposed_mean, total_information=proposed_info, total_covariance=proposed_covar)
                    temp_phi += x_k.expect_scalar

                self.new_cost = temp_phi + (0.5 * np.linalg.slogdet(D)[1])
                
                if self.debug:
                    print("--------------------------------")
                    print(f"| Backtrack #{backtrack_count}, alpha: {alpha}   |")
                    print(f"| New Cost:{self.new_cost}, Old Cost: {self.cur_cost} |")
                    print("--------------------------------")
                
                if alpha <= 0.1:
                    print("--------------------------------")
                    print("| Backtracking failed to find suitable step size | ")
                    print(f"|  Converged in {n_iters} iterations!  |")
                    print("--------------------------------")
                    for x_k in self.factored_states:
                        x_k.update_factor(total_mean=self.mean, total_information=self.information.copy(), total_covariance=self.covariance.copy())
                    return
                alpha *= 0.95
                    
            # L, D, _ = scipy.linalg.ldl(new_information)
            self.information = proposed_info
            self.covariance = proposed_covar
            self.mean = proposed_mean

            for x_k in self.factored_states:
                x_k.update_factor(total_mean=self.mean.copy(), total_information=self.information.copy(), total_covariance=self.covariance.copy())

            n_iters += 1
            self.cur_cost = self.new_cost

            if self.debug:
                # print( "########################################")
                # print("Iteration: ", n_iters)
                # print( "----------------------------------------")
                # print(f"Cost: ", self.cur_cost)
                # print(f"|delta_mu_{n_iters}|:  {size_mu}")
                # print(f"|delta_info_{n_iters}|: {size_info}")
                # # print(f"Info_{n_iters}: \n", self.information)
                # print( "########################################\n")
                print(f"Iter: {n_iters} || Cost: {self.cur_cost} || Step size (mu): {size_mu} || Step size (info): {size_info}")
            
    def backtrack(self, delta_mu:np.ndarray, new_info:np.ndarray, delta_info:np.ndarray, alpha=0.95, max_iters=10):
        prev_info = new_info.copy()
        prev_mean = self.mean.copy() + delta_mu
        prev_cost = self.cur_cost
        backtrack_iters = 0
        while(True):
            
            proposed_info = force_PSD(self.information + (alpha*delta_info))
            proposed_info = self._force_sparsity(proposed_info, self.factor_dof)
            L, D, _ = scipy.linalg.ldl(proposed_info, lower=True)
            proposed_covar = self.compute_covariance(L,D)
            proposed_mean = self.mean + (alpha*delta_mu)

            temp_phi = 0
            for x_k in self.factored_states:
                x_k.update_factor(total_mean=proposed_mean, total_information=proposed_info, total_covariance=proposed_covar)
                temp_phi += x_k.expect_scalar

            self.new_cost = temp_phi + (0.5 * np.linalg.slogdet(D)[1])
            
            if self.new_cost > prev_cost:
                break
            
            prev_cost = self.new_cost
            prev_mean = proposed_mean.copy()
            prev_info = proposed_info.copy()
            alpha *= 0.95
            backtrack_iters += 1
            
            if backtrack_iters == max_iters:
                break
        
        return prev_mean, prev_info

    
    def get_estimate_list(self):
        est_list = []
        for x_k in self.factored_states:
            x_k:FactoredState
            mean = x_k.get_mean()
            covar = x_k.get_covariance()
            stamp = x_k.stamp
            state_k = VectorState(value=mean, stamp=stamp)
            est_k = StateWithCovariance(state=state_k, covariance=covar)
            est_list.append(est_k)
        
        return est_list
    
    def compute_covariance(self, L, D):
        S = np.zeros_like(D)
        n = D.shape[0]
        # Iterate backwards over rows
        for j in range(n - 1, -1, -1):  # k = K, K-1, ..., 0
            # Diagonal element
            S[j, j] = 1 / D[j, j]
            # Iterate across columns
            for k in range(j, -1, -1):  # j = k, k-1, ..., 0
                for ell in range(k+1, n):
                    S[j, k] -= S[j, ell] * L[ell, k]
                S[k, j] = S[j, k]  # Symmetric matrix
        return S

    
    def _force_sparsity(self, info_matrix:np.ndarray, deg:int):
        sparse_matrix = np.triu(np.tril(info_matrix.copy(), k=(2*deg-1)), k=-(2*deg-1))
        return sparse_matrix


# %%

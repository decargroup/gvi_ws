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
from util.psd import force_PSD, force_sym, regularize
import matplotlib.pyplot as plt
from scipy.linalg import block_diag
from abc import abstractmethod

class GVI:
    def __init__(self, factored_states:List[FactoredState], total_dim:int, backtrack_on = True, debug=False):
        self.factored_states = factored_states
        self.total_dim = total_dim
        self.debug = debug
        self.backtrack_on = backtrack_on
        self.factor_dof = 0
        # Initialize mean and information
        self.mean = np.zeros((total_dim, 1))
        self.information = np.zeros((total_dim, total_dim))
        self.covariance = np.zeros((total_dim, total_dim))
        self.cur_cost = np.inf

        # Build full mean and information
        k = 0
        
        for x_k in self.factored_states:
            
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
        
        self.covariance = force_PSD(scipy.linalg.pinv(self.information))

        # Update states accordingly
        phi = np.zeros((1,1))
        for x_k in self.factored_states:
            x_k.update_factor(total_mean=self.mean, total_information=self.information, total_covariance=self.covariance)
            phi += x_k.expect_scalar
        self.cur_cost = phi.copy() + (0.5 * np.linalg.slogdet(self.information)[1])

                
    def solve(self):
        n_iters = 0
        if self.debug:
            print(f"Starting Cost: {self.cur_cost}")
        while(True):
            n_iters += 1
            # TODO: Implement backtracking
            # Update info
            phi_dx = np.zeros_like(self.mean)
            self.new_information = np.zeros_like(self.information)
            expect_matrix = np.zeros_like(self.information)
            prev_phi = np.zeros((1,1))
            for x_k in self.factored_states:
                # if isinstance(x_k, MeasurementFactor):
                    # print('Measurement')
                proj_k = x_k.projection
                prev_phi += x_k.expect_scalar
                phi_dx += proj_k.T @ x_k.phi_dx()
                self.new_information += proj_k.T @ x_k.phi_dx_dx() @ proj_k
                expect_matrix += proj_k.T @ x_k.expect_matrix @ proj_k
                
            # Force Sparsity and LDL decomposition
            # self.new_information = self._force_sparsity(self.new_information, deg=self.factor_dof)
            self.new_information = force_PSD(self.new_information)
            self.new_information = regularize(self.new_information)
            L, D, _ = scipy.linalg.ldl(self.new_information, lower=True)
            
            # self.new_covariance = force_PSD(self.compute_covariance(L, D))
            self.new_covariance = force_PSD(scipy.linalg.pinv(self.new_information))
            # This has been a bit stabler
            # delta_mu, _, _, _ = np.linalg.lstsq(self.new_information, -phi_dx, rcond=None)
            delta_mu = scipy.linalg.solve(self.new_information, -phi_dx)

            # expect_matrix = force_PSD(expect_matrix)
            # functional_dinfo = (-0.5*expect_matrix) + (0.5*self.covariance*prev_phi) + (0.5*self.covariance)
            # functional_dinfo = force_sym(functional_dinfo)
            # delta_info = -2 * self.information @ functional_dinfo @ self.information
            
            delta_info = self.new_information - self.information
            delta_info = force_sym(delta_info)
                    
            # Calculate breaking condition
            size_mu = np.linalg.norm(delta_mu)
            size_info = np.linalg.norm(delta_info)
            # size_info = np.linalg.trace(self.covariance @ delta_info @ self.covariance @ delta_info)
            

            # size_info = np.linalg.norm(new_information @ self.covariance)
            if n_iters > 10:
                print(f"Reached max iterations")
                print("|Info|: ", size_info)
                print("|mu|: ", size_mu)
                break

            self.new_phi = np.zeros((1,1))
            self.new_mean = self.mean + delta_mu
            for x_k in self.factored_states:
                # Update factor recomputes expectations as well
                x_k.update_factor(total_mean=self.new_mean.copy(), total_information=self.new_information.copy(), total_covariance=self.new_covariance.copy())
                # Update new phi, for new cost
                self.new_phi += x_k.expect_scalar

            self.new_cost = self.new_phi + (0.5 * np.linalg.slogdet(self.new_information)[1])
            delta_cost = (self.cur_cost - self.new_cost)
            if size_mu < 1e-8 and size_info<1e-6:
                print("--------------------------------")
                print(f"Iter: {n_iters} || Cost: {self.cur_cost} || Step size (mu): {size_mu} || Step size (info): {size_info}")
                print(f"|  Converged in {n_iters} iterations!  |")
                print("--------------------------------")
                for x_k in self.factored_states:
                    x_k.update_factor(total_mean=self.mean, total_information=self.information, total_covariance=self.covariance)
                break

            if np.abs(delta_cost / self.cur_cost) < 1e-6:
                print("--------------------------------")
                print(f"Iter: {n_iters} || Cost: {self.cur_cost} || Step size (mu): {size_mu} || Step size (info): {size_info}")
                print(f"|  Converged in {n_iters} iterations!  |")
                print("--------------------------------")
                for x_k in self.factored_states:
                    x_k.update_factor(total_mean=self.mean, total_information=self.information, total_covariance=self.covariance)
                break
            
            if self.new_cost >= self.cur_cost:
                if self.backtrack_on:
                    print(f"Starting backtracking as {self.new_cost} > {self.cur_cost}")
                    backtrack_success = self.backtrack(delta_mu, delta_info, alpha=1.0, max_iters=25)
                    if not backtrack_success:
                        print(f"Backtracking failed to return a suitable step size")
                        print("Exiting...")
                        print(f"Iter: {n_iters} || Cost: {self.cur_cost} || Step size (mu): {size_mu} || Step size (info): {size_info}")
                        return 
                else:
                    print(f"Exiting, didn't reduce cost from {self.new_cost} to {self.cur_cost}")
                    for x_k in self.factored_states:
                        x_k.update_factor(total_mean=self.mean, total_information=self.information, total_covariance=self.covariance)
                    self.new_covariance = self.covariance.copy()
                    self.new_information = self.information.copy()
                    self.new_mean = self.mean.copy()
                    self.new_cost = self.cur_cost
                    return

                
            
            # Update for next iteration
            self.information = self.new_information.copy()
            self.covariance = self.new_covariance.copy()
            self.mean = self.new_mean.copy()
            self.cur_cost = self.new_cost

            if self.debug:
                print(f"Iter: {n_iters} || Cost: {self.cur_cost} || Step size (mu): {size_mu} || Step size (info): {size_info}")
            
    def backtrack(self, delta_mu:np.ndarray, delta_info:np.ndarray, alpha=0.95, max_iters=10):
        backtrack_iters = 0
        while(True):
            proposed_info = self.information + (alpha*delta_info)
            # proposed_info = self._force_sparsity(proposed_info, self.factor_dof)
            proposed_info = force_PSD(proposed_info)
            proposed_info = regularize(proposed_info)
            L, D, _ = scipy.linalg.ldl(proposed_info, lower=True)
            # proposed_covar = force_PSD(self.compute_covariance(L,D))
            proposed_covar = force_PSD(scipy.linalg.inv(proposed_info))
            proposed_mean = self.mean + (alpha*delta_mu)
            temp_phi = 0
            for x_k in self.factored_states:
                x_k.update_factor(total_mean=proposed_mean, total_information=proposed_info, total_covariance=proposed_covar)
                temp_phi += x_k.expect_scalar

            self.new_cost = temp_phi + (0.5 * np.linalg.slogdet(proposed_info)[1])
            if backtrack_iters % 5 == 0:
                print(f"Backtrack #{backtrack_iters} || Alpha: {alpha} || Cost: {self.new_cost}")
            
            if (self.new_cost < self.cur_cost):
                # Backtracking Succeeded
                self.information = proposed_info.copy()
                self.covariance = proposed_covar.copy()
                self.mean = proposed_mean
                return True
            
            alpha = 0.95**backtrack_iters
            backtrack_iters += 1
            
            if backtrack_iters >= max_iters:
                # Backtrack Failed, Convergence?
                # Reupdate states with original values
                for x_k in self.factored_states:
                    x_k.update_factor(total_mean=self.mean, total_information=self.information, total_covariance=self.covariance)
                self.new_covariance = self.covariance.copy()
                self.new_information = self.information.copy()
                self.new_mean = self.mean.copy()
                self.new_cost = self.cur_cost
                return False

    
    def get_estimate_list(self):
        est_list = []
        stamp_list = []
        for x_k in self.factored_states:
            x_k:FactoredState
            mean = x_k.get_mean()
            covar = x_k.get_covariance()
            stamp = x_k.stamp
            state_k = VectorState(value=mean, stamp=stamp)
            est_k = StateWithCovariance(state=state_k, covariance=covar)
            if stamp not in stamp_list:
                est_list.append(est_k)
                stamp_list.append(stamp)
            else:
                idx = nav.find_nearest_stamp_idx(stamp_list, stamp)
                if (not np.allclose(mean.reshape((-1,1)), est_list[idx].state.value.reshape((-1,1)))) or (not np.allclose(covar, est_list[idx].covariance)):
                    print(x_k, "at time: ", stamp)
                    print(x_k.projection)
                    print(mean.ravel(), est_list[idx].state.value.ravel())
                    print(covar, "\n", est_list[idx].covariance)
                    raise ValueError("Mean and covariance don't match")
        
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

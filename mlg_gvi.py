# %%
import numpy as np
import scipy.linalg
import navlie as nav
from typing import Callable, Optional, List
from util.cubatures import gh_cubature, spherical_cubature
from mlg_factors import FactoredState, PriorFactor, ProcessFactor, MeasurementFactor, LandmarkPriorFactor
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
    def __init__(self, factored_states:List[FactoredState], total_dim:int, backtrack_on = True, debug=False, max_iters=10, backtrack_iters=5, init_alpha=1.0):
        # Params
        self.factored_states = factored_states
        self.total_dim = total_dim
        self.debug = debug
        self.backtrack_on = backtrack_on
        self.backtrack_iters = backtrack_iters
        self.max_iters = max_iters
        self.factor_dof = factored_states[0].dof
        self.init_alpha = init_alpha

        # Initialize mean and information
        self.mean = np.zeros((total_dim, 1), dtype=np.float64)
        self.information = np.zeros((total_dim, total_dim), dtype=np.float64)
        self.covariance = np.zeros((total_dim, total_dim), dtype=np.float64)
        self.cur_cost = np.inf
        self.last_alpha = None

        # Build full mean and information
        k = 0
        for x_k in self.factored_states:
            
            if isinstance(x_k, PriorFactor):
                # self.factor_dof = x_k.dof
                self.mean[k:k+x_k.dof] = x_k.get_mean_vector()
                self.information[k:k+x_k.dof, k:k+x_k.dof] = x_k.get_information()
                k+= x_k.dof
            
            if isinstance(x_k, ProcessFactor):
                dof = x_k.dof
                self.mean[k:k+dof] = x_k.get_mean_vector()
                self.information[k:k+dof, k:k+dof] = x_k.get_information()
                self.information[k-dof:k, k:k+dof] = x_k.get_cross_information()
                self.information[k:k+dof, k-dof:k] = x_k.get_cross_information().T
                k += dof
                
        self.information = force_PSD(self.information)

        # TODO: Fix the sparsity of computing covariance
        # L, D, _ = scipy.linalg.ldl(self.information, lower=True)
        # self.covariance = force_PSD(self.compute_covariance(L, D))
        self.covariance = force_PSD(scipy.linalg.pinv(self.information))

        # Update states accordingly
        phi = np.zeros((1,1))
        for x_k in self.factored_states:
            x_k.update_factor(total_mean=self.mean, total_information=self.information, total_covariance=self.covariance, delta_mean=np.zeros_like(self.mean))
            phi += x_k.expect_scalar
        self.cur_cost = phi + (0.5 * np.linalg.slogdet(self.information)[1])

                
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
            prev_phi = 0.0
            for x_k in self.factored_states:
                proj_k = x_k.total_projection
                prev_phi += x_k.expect_scalar
                phi_dx += proj_k.T @ x_k.phi_dx()
                self.new_information += proj_k.T @ x_k.phi_dx_dx() @ proj_k
                if isinstance(x_k, LandmarkPriorFactor):
                    print("Prev Values: ")
                    print(x_k.covariance)
                    print(x_k.mean)
                
            # Force sparsity, PSD, regularize
            self.new_information = self._force_sparsity(self.new_information, deg=self.factor_dof)
            self.new_information = force_PSD(self.new_information)
            self.new_information = regularize(self.new_information)

            # TODO: Fix sparsity
            # L, D, _ = scipy.linalg.ldl(self.new_information, lower=True)
            # self.new_covariance = force_PSD(self.compute_covariance(L, D))
            
            # Compute Covariance
            # self.new_covariance = force_PSD(scipy.linalg.inv(self.new_information))
            self.new_covariance = self._force_sparsity(scipy.linalg.inv(self.new_information), deg=self.factor_dof)
            self.new_covariance = force_PSD(self.new_covariance)

            # Solve for mean update step
            # This has been a bit stabler
            delta_mu = scipy.linalg.solve(self.new_information, -phi_dx)
            self.new_mean = self.mean + delta_mu

            # Compute delta_info
            delta_info = self.new_information - self.information
            delta_info = regularize(delta_info)
            delta_info = force_sym(delta_info)
                    
            # Calculate breaking condition
            size_mu = np.linalg.norm(delta_mu)
            # size_info = np.abs(np.linalg.slogdet(delta_info)[1])
            sqrt_c2 = scipy.linalg.sqrtm(self.new_covariance)
            # Maybe use this for delta info somehow?
            size_info = np.linalg.norm(self.covariance + self.new_covariance - 2*scipy.linalg.sqrtm(sqrt_c2 @ self.covariance @ sqrt_c2), 'fro')
            # Update factors with new mean, covariance, and info
            self.new_phi = 0.0
            for x_k in self.factored_states:
                # Update factor recomputes expectations as well
                x_k.update_factor(total_mean=np.copy(self.new_mean), total_information=np.copy(self.new_information), total_covariance=np.copy(self.new_covariance), delta_mean=np.copy(delta_mu))
                # Update new phi, for new cost
                self.new_phi += x_k.expect_scalar
                if isinstance(x_k, LandmarkPriorFactor):
                    print("New State: ")
                    print(x_k.covariance)
                    print(x_k.mean)

            # Compute cost at this update
            self.new_cost = self.new_phi + (0.5 * np.linalg.slogdet(self.new_information)[1])
            # Convergence Tests
            # TODO: Put this into convergence() function
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

            if n_iters >= self.max_iters:
                for x_k in self.factored_states:
                    x_k.update_factor(total_mean=self.mean, total_information=self.information, total_covariance=self.covariance)
                print(f"Reached max iterations")
                print("|Info|: ", size_info)
                print("|mu|: ", size_mu)
                break

            # Backtracking Loops
            if self.new_cost >= self.cur_cost:
                if self.backtrack_on:
                    print(f"Starting backtracking as {self.new_cost} > {self.cur_cost}")
                    backtrack_success = self.backtrack(np.copy(delta_mu), np.copy(delta_info), max_iters=self.backtrack_iters, alpha=self.init_alpha)
                    if backtrack_success:
                        self.mean, self.information, self.covariance = self.update_global_vars()
                    else:
                        print(f"Backtracking failed to return a suitable step size")
                        print("Exiting...")
                        print(f"Iter: {n_iters} || Cost: {self.cur_cost} || Step size (mu): {size_mu} || Step size (info): {size_info}")
                        return 
                else:
                    print(f"Exiting, didn't reduce cost from {self.new_cost} to {self.cur_cost}")
                    for x_k in self.factored_states:
                        x_k.update_factor(total_mean=self.mean, total_information=self.information, total_covariance=self.covariance)
                        if isinstance(x_k, LandmarkPriorFactor):
                            print("Returning to previous: ")
                            print(x_k.covariance)
                            print(x_k.mean)

                    return
            
            # Update for next iteration
            self.mean = self.new_mean.copy()
            self.information = self.new_information.copy()
            self.covariance = self.new_covariance.copy()
            self.mean, self.information, self.covariance = self.update_global_vars()
            # self.mean, _, _ = self.update_global_vars()
            # self.information = np.copy(self.new_information)
            # self.covariance = np.copy(self.new_covariance)
            self.cur_cost = np.copy(self.new_cost)
            
            if self.debug:
                print(f"Iter: {n_iters} || Cost: {self.cur_cost} || Step size (mu): {size_mu} || Step size (info): {size_info}")
            
    def backtrack(self, delta_mu:np.ndarray, delta_info:np.ndarray, max_iters=10, alpha = 1.0):
        alpha = alpha if self.last_alpha is None else self.last_alpha
        backtrack_iters = 0
        while(True):
            proposed_info = self.information + (alpha*delta_info)
            # proposed_info = self._force_sparsity(proposed_info, self.factor_dof)
            proposed_info = force_PSD(proposed_info)
            proposed_info = regularize(proposed_info)
            # L, D, _ = scipy.linalg.ldl(proposed_info, lower=True)
            # proposed_covar = force_PSD(self.compute_covariance(L,D))
            proposed_covar = force_PSD(scipy.linalg.inv(proposed_info))
            proposed_mean = self.mean + (alpha*delta_mu)
            temp_phi = 0.0
            for x_k in self.factored_states:
                x_k.update_factor(total_mean=proposed_mean, total_information=proposed_info, total_covariance=proposed_covar, delta_mean=alpha*delta_mu)
                temp_phi += x_k.expect_scalar

            self.new_cost = temp_phi + (0.5 * np.linalg.slogdet(proposed_info)[1])
            if backtrack_iters % 5 == 0:
                print(f"Backtrack #{backtrack_iters} || Alpha: {alpha} || Cost: {self.new_cost}")
            
            if (self.new_cost < self.cur_cost):
                # Backtracking Succeeded
                self.new_information = np.copy(proposed_info)
                self.new_covariance = np.copy(proposed_covar)
                self.new_mean = np.copy(proposed_mean)
                self.last_alpha = alpha
                return True
            
            alpha *= 0.95
            backtrack_iters += 1
            
            if backtrack_iters >= max_iters or alpha <= 1e-9:
                # Backtrack Failed, Convergence?
                # Reupdate states with original values
                for x_k in self.factored_states:
                    x_k.update_factor(total_mean=self.mean, total_information=self.information, total_covariance=self.covariance)
                self.new_covariance = np.copy(self.covariance)
                self.new_information = np.copy(self.information)
                self.new_mean = np.copy(self.mean)

                self.new_cost = np.copy(self.cur_cost)
                return False

    
    def get_estimate_list(self, get_landmark=False):
        est_list = []
        landmark_est:List[StateWithCovariance] = []
        stamp_list = []
        for x_k in self.factored_states:
            x_k:FactoredState

            state_k = x_k.get_mean()
            covar = x_k.get_covariance()
            stamp = x_k.stamp
            est_k = StateWithCovariance(state=state_k, covariance=covar)
            if get_landmark:
                if state_k.state_id[0] =='l':
                    landmark_est.append(est_k)
                    
            if stamp is None:
                pass
            elif stamp not in stamp_list:
                est_list.append(est_k)
                stamp_list.append(stamp)
            else:
                idx = nav.find_nearest_stamp_idx(stamp_list, stamp)
                
                if (not np.allclose(state_k.value, est_list[idx].state.value)) or (not np.allclose(covar, est_list[idx].covariance)):
                    print(x_k, "at time: ", stamp)
                    print(x_k.projection)
                    print(state_k.value, est_list[idx].state.value)
                    print(covar, "\n", est_list[idx].covariance)
                    raise ValueError("Mean and covariance don't match")
        if get_landmark:
            return est_list, landmark_est
        
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
    
    def from_map(self, map_covariance):
        self.covariance = force_PSD(np.copy(map_covariance))
        self.information = self._force_sparsity(force_PSD(scipy.linalg.inv(np.copy(map_covariance))), self.factor_dof)
        # Update states accordingly
        phi = np.zeros((1,1))
        for x_k in self.factored_states:
            x_k.update_factor(total_mean=self.mean, total_information=self.information, total_covariance=self.covariance)
            phi += x_k.expect_scalar
        self.cur_cost = phi + (0.5 * np.linalg.slogdet(self.information)[1])

    def update_global_vars(self):
        k = 0
        new_mean = np.zeros_like(self.mean)
        new_covar = np.zeros_like(self.covariance)
        new_info = np.zeros_like(self.information)

        for x_k in self.factored_states:

            if isinstance(x_k, PriorFactor):
                new_mean[k:k+x_k.dof] = x_k.get_mean_vector()
                new_info[k:k+x_k.dof, k:k+x_k.dof] = x_k.get_information()
                new_covar[k:k+x_k.dof, k:k+x_k.dof] = x_k.get_covariance()
                k+= x_k.dof
            
            if isinstance(x_k, ProcessFactor):
                dof = x_k.dof
                new_mean[k:k+dof] = x_k.get_mean_vector()
                new_info[k:k+dof, k:k+dof] = x_k.get_information()
                new_info[k-dof:k, k:k+dof] = x_k.get_cross_information()
                new_info[k:k+dof, k-dof:k] = x_k.get_cross_information().T
                new_covar[k:k+dof, k:k+dof] = x_k.get_covariance()
                new_covar[k-dof:k, k:k+dof] = x_k.get_cross_covariance()
                new_covar[k:k+dof, k-dof:k] = x_k.get_cross_covariance().T
                k += dof
        return new_mean, new_info, new_covar

    def _force_sparsity(self, info_matrix:np.ndarray, deg:int):
        sparse_matrix = np.triu(np.tril(info_matrix.copy(), k=(2*deg-1)), k=-(2*deg-1))
        return sparse_matrix


# %%

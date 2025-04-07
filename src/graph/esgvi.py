import collections.abc
import time
from typing import Dict, Hashable, List, Tuple

import numpy as np
import navlie as nav
from scipy import sparse
from scipy import linalg as splg

from src.graph.factors import Factor
from src.util.psd import force_sym_PSD, force_sym, regularize
from navlie.types import State
from navlie.lib import MatrixLieGroupState


class ESGVI:
    """Main class for ESGVI FactorGraph"""

    def __init__(
        self,
        max_iters: int = 10,
        step_tol: float = 1e-6,
        backtrack: bool = True,
        backtrack_iters: int = 10,
        init_step_distance: float = 1.0,
        verbose: bool = True,
    ):
        # Solver Params
        self.max_iters = max_iters
        self.step_tol = step_tol

        self.verbose = verbose

        # Initial values of states
        self.init_states: Dict[str, State] = {}
        # Dict of previous state values for backtracking purposes
        self._prev_states: Dict[str, State] = {}
        # Dict of all current values of the states
        self.states: Dict[str, State] = {}
        self.state_slices: Dict[str, slice] = {}

        # List of all factors in ESGVI graph
        self.factor_list: List[Factor] = []
        self.factor_slices: List[slice] = []

        # Size of the problem
        self._graph_total_dof: int = None
        self._num_states: int = None
        self._num_landmarks: int = None
        self._size_factors: int = None

        # Cost values
        self.new_cost: int = None
        self.prev_cost: int = None
        self.cost_history: List[int] = []

        # Values
        self._delta_mean: np.ndarray = None
        self._information_matrix: np.ndarray = None
        self._covariance_matrix: np.ndarray = None

        # Backtracking Values
        self.backtrack = backtrack
        self.backtrack_iters = backtrack_iters
        self.init_step_distance = init_step_distance
        self.last_alpha = None

    def is_converged(self, delta_mean, new_info, new_covar):
        # TODO: Check if this should cur_cost or prev_cost
        cost = self.prev_cost
        delta_cost = self.new_cost - self.prev_cost
        # Delta mean size
        size_delta_mean = np.linalg.norm(delta_mean)
        # Delta info size
        delta_info = new_info - self._information_matrix
        sqrt_c2 = splg.sqrtm(new_covar)
        size_delta_info = np.linalg.norm(
            self._covariance_matrix
            + new_covar
            - 2 * (splg.sqrtm(sqrt_c2 @ self._covariance_matrix @ sqrt_c2)),
            "fro",
        )
        converged = False
        if size_delta_mean <= self.step_tol and size_delta_info <= self.step_tol:
            converged = True
            print(f"Mean and covariance converged to {self.step_tol} tolerance.")

        if delta_cost is not None:
            if cost != 0:
                rel_cost_change = delta_cost / cost
                if rel_cost_change <= self.step_tol:
                    converged = True
                    print(f"Coverged with relative cost change of {self.step_tol}.")

        return converged

    def add_factor(self, factor: Factor):
        """Adds a factor to the ESGVI Factor Graph."""

        # If the user gives a list of factors, extend the existing list
        if isinstance(factor, list):
            self.factor_list.extend(factor)

        else:
            self.factor_list.append(factor)

    def add_state(self, key: Hashable, variable: State):
        """Adds a state to the ESGVI Factor Graph."""
        if self._graph_total_dof is None:
            self._graph_total_dof = 0

        self.init_states[key] = variable
        self.state_slices[key] = slice(
            self._graph_total_dof, self._graph_total_dof + variable.dof
        )
        self._graph_total_dof += variable.dof

        if key[0] == "x":
            self._num_states += 1

        elif key[0] == "l":
            self._num_landmarks += 1

        else:
            raise ValueError("State keys must start with either 'x' or 'l'.")

    def solve(self) -> Dict[Hashable, State]:
        """Solves the optimization problem."""
        self.states = {k: v.copy() for k, v in self.init_states.items()}
        self._prev_states = {k: v.copy() for k, v in self.init_states.items()}
        while True:
            # Calculate New Information, Phi_dx
            new_info = np.zeros_like(self._information_matrix)
            phi_dx = np.zeros_like(self._delta_mean)
            self.prev_cost = 0.5 * np.linalg.slogdet(self._information_matrix)[1]

            for factor in self.factor_list:
                cost_update, phi_dx_update, info_update = factor.evaluate_derivatives(
                    self.states, self._covariance_matrix, self._information_matrix
                )
                self.prev_cost += cost_update
                phi_dx += phi_dx_update
                new_info += info_update

            new_info = force_sym_PSD(new_info)
            if np.linalg.cond(new_info) < (1 / np.finfo(new_info.dtype).eps):
                new_info = regularize(new_info)

            # TODO: Redo this using sparsity tricks
            new_covar = force_sym_PSD(splg.pinv(new_info))
            self._delta_mean = splg.solve(new_info, -phi_dx)
            new_info, new_covar = self.update_states(
                self._delta_mean, new_info, new_covar
            )
            self.new_cost = self.calculate_new_cost(new_info, new_covar)
            if self.is_converged(self._delta_mean, new_info, new_covar):
                return self.states

            if self.new_cost < self.prev_cost:
                self._information_matrix = new_info
                self._covariance_matrix = new_covar
                self._prev_states = {k: v.copy() for k, v in self.states.items()}
                self.cost_history.append(self.new_cost)
                self.prev_cost = self.new_cost
            else:
                backtrack_success, new_info, new_covar = self.backtrack(
                    new_info, new_covar, init_step_dist=self.init_step_distance
                )
                if backtrack_success:
                    self._information_matrix = new_info
                    self._covariance_matrix = new_covar
                    self._prev_states = {k: v.copy() for k, v in self.states.items()}
                    self.cost_history.append(self.new_cost)
                    self.prev_cost = self.new_cost
                else:
                    print(f"Backtracking failed...")
                    return self._prev_states

    def update_states(
        self,
        delta_mean: np.ndarray,
        new_information: np.ndarray,
        new_covariance: np.ndarray,
        update_covariance: bool = True,
    ):
        for key, state in self.states.items():
            state_slice = self.state_slices[key]
            state_delta_mean = delta_mean[state_slice, 0]
            state.plus(state_delta_mean)
            # Change the covariance and information values if on-manifold
            if update_covariance and isinstance(state, MatrixLieGroupState):
                if state.direction == "left":
                    jac = state.group.left_jacobian(state_delta_mean)
                    jac_inv = state.group.left_jacobian_inv(state_delta_mean)
                if state.direction == "right":
                    jac = state.group.right_jacobian(state_delta_mean)
                    jac_inv = state.group.right_jacobian_inv(state_delta_mean)

                new_state_covar = new_covariance[state_slice, state_slice].copy()
                new_state_info = new_information[state_slice, state_slice].copy()

                new_state_covar = jac_inv @ new_state_covar @ jac_inv.T
                new_state_info = jac.T @ new_state_info @ jac

                new_covariance[state_slice, state_slice] = new_state_covar
                new_information[state_slice, state_slice] = new_state_info

        return new_information, new_covariance

    def calculate_new_cost(self, new_information, new_covariance):
        new_cost = 0.5 * np.linalg.slogdet(new_information)[1]
        for factor in self.factor_list:
            new_cost += factor.evaluate_factor_cost(
                self.states, new_covariance, new_information
            )

        return new_cost

    def backtrack(
        self, new_info: np.ndarray, new_covar: np.ndarray, init_step_dist=1
    ) -> Tuple[bool, np.ndarray, np.ndarray]:
        # Recopy old states
        self.states = {k: v.copy() for k, v in self._prev_states.items()}
        delta_info = force_sym(new_info - self._information_matrix)
        delta_covar = force_sym(new_covar - self._covariance_matrix)
        # TODO: Try this for faster as inverse is linear
        # delta_covar = force_sym_PSD(new_covar -self._covariance_matrix)

        # Backtracking values
        alpha = init_step_dist if self.last_alpha is None else self.last_alpha
        backtrack_delta_mean = alpha * self._delta_mean
        backtrack_info = (alpha * delta_info) + self._information_matrix
        backtrack_covar = (alpha * delta_covar) + self._covariance_matrix

        n_iters = 0

        while n_iters < self.backtrack_iters:
            backtrack_info, backtrack_covar = self.update_states(
                delta_mean=backtrack_delta_mean,
                new_information=backtrack_info,
                new_covariance=backtrack_covar,
                update_covariance=False,
            )  # Don't update manifold covariance, this is accounted in the new_info subtraction

            # Calculate new cost with backtracked state, information and covariance values.
            backtrack_cost = 0.5 * np.linalg.slogdet(backtrack_info)[1]
            for factor in self.factor_list:
                backtrack_cost += factor.evaluate_factor_cost(
                    self.states, backtrack_covar, backtrack_info
                )

            if backtrack_cost < self.prev_cost:
                self.last_alpha = alpha
                self.new_cost = backtrack_cost
                return True, backtrack_info, backtrack_covar
            else:
                alpha *= 0.95
                backtrack_delta_mean = alpha * self._delta_mean
                backtrack_info = (alpha * delta_info) + self._information_matrix
                backtrack_covar = (alpha * delta_covar) + self._covariance_matrix

            n_iters += 1

        return False, backtrack_info, backtrack_covar

    def init_covariance(self, covariance: np.ndarray = None):

        for factor in self.factor_list:
            raise NotImplementedError

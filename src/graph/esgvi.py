import collections.abc
import time
from typing import Dict, Hashable, List, Tuple

import numpy as np
import navlie as nav
from scipy import sparse
from scipy import linalg as splg

from src.graph.factors import Factor
from src.util.psd import force_sym_PSD, force_sym, regularize, isPD
from src.util.sparsity import force_block_banded_sparsity
from navlie.types import State
from navlie.lib import MatrixLieGroupState

from scipy import sparse


class ESGVI:
    """Main class for ESGVI FactorGraph"""

    def __init__(
        self,
        max_iters: int = 10,
        step_tol: float = 1e-6,
        backtrack_on: bool = True,
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
        self._graph_total_dof: int = 0
        self._graph_pose_dof: int = 0
        self._graph_landmark_dof: int = 0
        self._num_poses: int = 0
        self._pose_dof: int = 0
        self._num_landmarks: int = 0

        # Cost values
        self.new_cost: int = None
        self.prev_cost: int = None
        self.cost_history: List[float] = []
        self.factor_cost_history: List[List[Tuple[str, float]]] = []

        # Values
        self._delta_mean: np.ndarray = None
        self._information_matrix: np.ndarray = None
        self._covariance_matrix: np.ndarray = None

        # Backtracking Values
        self.backtrack_on = backtrack_on
        self.backtrack_iters = backtrack_iters
        self.init_step_distance = init_step_distance
        self.last_alpha = None

    def is_converged(self, delta_mean, new_info, new_covar, n_iters):
        # TODO: Check if this should cur_cost or prev_cost
        cost = self.prev_cost
        delta_cost = np.abs(self.new_cost - self.prev_cost)
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
            if self.verbose:
                print(f"Mean and covariance converged to {self.step_tol} tolerance.")

        if delta_cost is not None:
            if cost != 0:
                rel_cost_change = delta_cost / cost
                if rel_cost_change <= self.step_tol:
                    converged = True
                    if self.verbose:
                        print(
                            f"Converged with relative cost change of {self.step_tol}."
                        )

        if n_iters >= self.max_iters:
            converged = True
            if self.verbose:
                print(f"Reached max iterations of {n_iters}.")

        if self.verbose:
            print(
                f"Iter: {n_iters} || Cost: {float(self.prev_cost):.8f} || Step Size (Mean): {size_delta_mean:.4e} || Step Size (Info): {size_delta_info:.4e} || dC/C: {rel_cost_change[0,0]:.4e}"
            )

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

        self.init_states[key] = variable
        self.state_slices[key] = slice(
            self._graph_total_dof, self._graph_total_dof + variable.dof
        )
        self._graph_total_dof += variable.dof

        if key[0] == "x":
            # Number of total poses
            self._num_poses += 1
            # Total dof of poses
            self._graph_pose_dof += variable.dof
            # Individual dof of pose
            self._pose_dof = variable.dof

        elif key[0] == "l":
            # Number of total landmarks
            self._num_landmarks += 1
            # Total dof of landmarks
            self._graph_landmark_dof += variable.dof

        else:
            raise ValueError("State keys must start with either 'x' or 'l'.")

    def solve(self) -> Dict[Hashable, State]:
        """Solves the optimization problem."""
        self.states = {k: v.copy() for k, v in self.init_states.items()}
        self._prev_states = {k: v.copy() for k, v in self.init_states.items()}
        init_cost = self.calculate_cost(
            self.init_states,
            information=self._information_matrix,
            covariance=self._covariance_matrix,
        )
        if self.verbose:
            print(f"Starting ESGVI with initial cost: {float(init_cost):.8f} ")
        n_iters = 0
        while True:
            # Calculate New Information, Phi_dx
            new_info = np.zeros_like(self._information_matrix)
            phi_dx = np.zeros_like(self._delta_mean)
            sign, logdet = np.linalg.slogdet(self._information_matrix)
            self.prev_cost = 0.5 * sign * logdet

            for factor in self.factor_list:
                factor_states = [self._prev_states[key].copy() for key in factor.keys]
                cost_update, phi_dx_update, info_update = factor.evaluate_derivatives(
                    factor_states, self._covariance_matrix, self._information_matrix
                )
                self.prev_cost += cost_update
                phi_dx += phi_dx_update
                new_info += info_update

            # Force sparsity constraints of information matrix
            # new_info = self._force_sparsity(new_info)
            # Force symmetric PSD
            # new_info = force_sym_PSD(new_info)

            # TODO: Redo this using sparsity trick
            new_covar = force_sym_PSD(splg.inv(new_info))

            self._delta_mean = splg.solve(new_info, -phi_dx)
            self.states, new_info, new_covar = self.update_states(
                self._delta_mean, self._prev_states, new_info, new_covar
            )
            assert np.all(new_info == new_info.T), "New Info is not symmetric"
            assert np.all(new_covar == new_covar.T), "New Info is not symmetric"

            self.new_cost = self.calculate_cost(self.states, new_info, new_covar)

            if self.is_converged(self._delta_mean, new_info, new_covar, n_iters):
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
                    return self._prev_states
            n_iters += 1

    def update_states(
        self,
        delta_mean: np.ndarray,
        old_states: Dict[str, State],
        information: np.ndarray,
        covariance: np.ndarray,
        update_covariance: bool = True,
    ) -> Tuple[Dict[str, State], np.ndarray, np.ndarray]:
        """Updates each of the states, and associated information/covariances using the individual "plus" operators
        of each state.
        Parameters
        ----------
        delta_mean : np.ndarray
            Increment to the entire batch state.
        old_states : Dict[Hashable, State], optional
            States to be updated.
        information: np.ndarray
            Information matrix of the batch state.
        covariance: np.ndarray
            Covariance matrix of the batch state.
        Return
        ----------
        Tuple(Dict(str, state), np.ndarray, np.ndarray)
            Tuple describing the updated state dictionary, information and covariance matrices.
        """
        new_states = {k: v.copy() for k, v in old_states.items()}
        for key, state in old_states.items():
            state_slice = self.state_slices[key]
            state_delta_mean = delta_mean[state_slice, 0]
            new_states[key] = state.plus(state_delta_mean)
            # Change the covariance and information values if on-manifold
            if update_covariance and isinstance(state, MatrixLieGroupState):
                if state.direction == "left":
                    jac = state.group.left_jacobian(state_delta_mean)
                    jac_inv = state.group.left_jacobian_inv(state_delta_mean)
                if state.direction == "right":
                    jac = state.group.right_jacobian(state_delta_mean)
                    jac_inv = state.group.right_jacobian_inv(state_delta_mean)

                new_state_covar = covariance[state_slice, state_slice].copy()
                new_state_info = information[state_slice, state_slice].copy()

                # TODO: Should this be forced PSD?
                # new_state_covar = jac_inv @ new_state_covar @ jac_inv.T
                # new_state_info = jac.T @ new_state_info @ jac
                # new_state_covar = force_sym_PSD(jac_inv @ new_state_covar @ jac_inv.T)
                # new_state_info = force_sym_PSD(jac.T @ new_state_info @ jac)

                covariance[state_slice, state_slice] = new_state_covar
                information[state_slice, state_slice] = new_state_info

        #  TODO: Check this
        covariance = force_sym_PSD(covariance)
        information = force_sym_PSD(information)

        return new_states, information, covariance

    def calculate_cost(
        self,
        new_states: Dict[str, State],
        information: np.ndarray,
        covariance: np.ndarray,
    ) -> np.ndarray:
        sign, logdet = np.linalg.slogdet(information)
        new_cost = 0.5 * sign * logdet
        factor_cost_list = []
        for factor in self.factor_list:
            factor_states = [new_states[key].copy() for key in factor.keys]
            factor_cost = factor.evaluate_factor_cost(
                factor_states, covariance, information
            )
            factor_cost_info = tuple([factor.type, factor.keys, factor_cost[0, 0]])
            factor_cost_list.append(factor_cost_info)

            new_cost += factor_cost

        self.factor_cost_history.append(factor_cost_list)
        return new_cost

    def backtrack(
        self,
        new_info: np.ndarray,
        new_covar: np.ndarray,
        init_step_dist: float = 1,
        alpha_multiplier: float = 0.9,
    ) -> Tuple[bool, np.ndarray, np.ndarray]:
        if self.verbose:
            print(
                f"Starting backtracking as {float(self.new_cost):.8f} > {float(self.prev_cost):.8f}"
            )
        delta_info = force_sym(new_info - self._information_matrix)
        delta_covar = force_sym(new_covar - self._covariance_matrix)
        # assert isPD(delta_info), "Delta Info is not positive-definite"
        # assert isPD(delta_covar), "Delta Covar is not positive-definite"
        # TODO: Try this for faster as inverse is linear
        # delta_covar = force_sym_PSD(new_covar -self._covariance_matrix)

        # Backtracking values
        alpha = init_step_dist if self.last_alpha is None else self.last_alpha
        backtrack_delta_mean = alpha * self._delta_mean
        backtrack_info = force_sym_PSD((alpha * delta_info) + self._information_matrix)
        backtrack_covar = force_sym_PSD((alpha * delta_covar) + self._covariance_matrix)
        assert isPD(backtrack_info), "Delta Info is not positive-definite"
        assert isPD(backtrack_covar), "Delta Covar is not positive-definite"
        n_iters = 0

        while n_iters < self.backtrack_iters:
            self._backtrack_states, backtrack_info, backtrack_covar = (
                self.update_states(
                    delta_mean=backtrack_delta_mean,
                    old_states=self._prev_states,
                    information=backtrack_info,
                    covariance=backtrack_covar,
                    update_covariance=False,
                )
            )  # Don't update manifold covariance, this is accounted in the new_info subtraction

            # Calculate new cost with backtracked state, information and covariance values.
            self._backtrack_cost = self.calculate_cost(
                self._backtrack_states, backtrack_info, backtrack_covar
            )

            n_iters += 1
            if self.verbose and (n_iters - 1) % 10 == 0:
                print(
                    f"Backtrack: {n_iters} || Step Size: {alpha:.4e} || Cost: {float(self._backtrack_cost):.8f}"
                )

            if self._backtrack_cost < self.prev_cost:
                self.last_alpha = alpha
                self.new_cost = self._backtrack_cost
                self.states = {k: v.copy() for k, v in self._backtrack_states.items()}
                return True, backtrack_info, backtrack_covar
            else:
                alpha *= alpha_multiplier
                backtrack_delta_mean = alpha * self._delta_mean
                backtrack_info = force_sym_PSD(
                    (alpha * delta_info) + self._information_matrix
                )
                backtrack_covar = force_sym_PSD(
                    (alpha * delta_covar) + self._covariance_matrix
                )

        if self.verbose:
            print(f"Backtracking didn't find suitable step size after {n_iters}")

        return False, backtrack_info, backtrack_covar

    def init_covariance(self, information: np.ndarray):
        if information.shape != (self._graph_total_dof, self._graph_total_dof):
            raise ValueError(
                f"Information has shape {information.shape}, rather than ({self._graph_total_dof}, {self._graph_total_dof})."
            )
        self._information_matrix = information.copy()
        self._covariance_matrix = splg.inv(self._information_matrix)
        self._delta_mean = np.zeros((self._graph_total_dof, 1))

    def get_covariance_block(self, key_1: Hashable, key_2: Hashable) -> np.ndarray:
        """Retrieve the covariance block corresponding to two variables.

        Parameters
        ----------
        key_1 : Hashable
            Key of first variable.
        key_2 : Hashable
            Key of second variable.

        Returns
        -------
        np.ndarray
            Covariance block corresponding to the two variables.
        """

        # Extract relevant block
        try:
            var_1_slice = self.state_slices[key_1]
            var_2_slice = self.state_slices[key_2]

            return self._covariance_matrix[var_1_slice, var_2_slice]
        except KeyError as e:
            print(f"Cannot compute covariance block!")

    def _force_sparsity(self, info_matrix: np.ndarray):
        info_matrix_cp = info_matrix.copy()
        # Force sparsity of the pose blocks
        new_pose_info_block = info_matrix_cp[
            0 : self._graph_pose_dof, 0 : self._graph_pose_dof
        ]
        info_matrix_cp[0 : self._graph_pose_dof, 0 : self._graph_pose_dof] = (
            force_block_banded_sparsity(new_pose_info_block, block_size=self._pose_dof)
        )
        return info_matrix_cp

import collections.abc
import time
from typing import Dict, Hashable, List, Tuple

import numpy as np
from scipy import sparse
from scipy import linalg as splg

from src.graph.factors import Factor
from navlie.types import State


class ESGVI:
    """Main class for ESGVI FactorGraph"""

    def __init__(
        self,
        max_iters: int = 10,
        step_tol: float = 1e-6,
        backtrack: bool = True,
        backtrack_iters: int = 10,
        init_alpha=1,
        verbose: bool = True,
    ):
        # Solver Params
        self.max_iters = max_iters
        self.step_tol = step_tol
        self.backtrack = backtrack
        self.backtrack_iters = backtrack_iters
        self.init_alpha = init_alpha
        self.verbose = verbose

        # Initial values of states
        self.init_states: Dict[str, State] = {}
        # Dict of all current values of the states
        self.states: Dict[str, State] = {}
        self.state_slices: Dict[str, slice] = {}

        # List of all factors in ESGVI graph
        self.factor_list: List[Factor] = []
        self.factor_slices: List[slice] = []

        # Size of the problem
        self._size_graph: int = None
        self._size_pose: int = None
        self._size_landmark: int = None
        self._size_factors: int = None

        # Cost values
        self.new_cost: int = None
        self.prev_cost: int = None
        self.cost_history: List[int] = []

        # Values
        self._mean: np.ndarray = None
        self._information_matrix: np.ndarray = None
        self._covariance_matrix: np.ndarray = None

    def is_converged(self, delta_mean, delta_info, delta_cost, cost):
        converged = False
        if delta_mean <= self.step_tol and delta_info <= self.step_tol:
            converged = True

        if delta_cost is not None:
            if cost != 0:
                rel_cost_change = delta_cost / cost
                if rel_cost_change <= self.step_tol:
                    converged = True

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

    def solve(self) -> Dict[Hashable, State]:
        """Solves the optimization problem."""
        self.states = {k: v.copy() for k, v in self.init_states.items()}
        pass

    def update_states(
        self,
        delta_mean: np.ndarray,
        new_information: np.ndarray,
        new_covariance: np.ndarray,
    ):
        pass

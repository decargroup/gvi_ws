import numpy as np
import scipy.linalg
from typing import Hashable, List, Tuple, Dict
import navlie as nav
from pymlg.numpy.se2 import SE2, SO2
from typing import Callable, Optional, List
from gvi_ws.util.cubatures import (
    gh_cubature,
    spherical_cubature,
    unscented_cubature,
    student_t_cubature,
    trans_spherical_cubature,
    trans_gh_cubature, 
    trans_unscented_cubature
)
from navlie.lib.states import (
    VectorState,
    SE2State,
    State,
    CompositeState,
    MatrixLieGroupState,
)
from navlie.types import ProcessModel, Measurement, Input, StateWithCovariance
from navlie.batch.problem import Problem
from navlie.utils import find_nearest_stamp_idx
from gvi_ws.util.psd import force_sym_PSD, isPD, force_sym
from abc import abstractmethod


class Factor:
    def __init__(
        self,
        keys: List[Hashable],
        variable_slices: Dict[str, slice],
        projection: np.ndarray,
        cubature: str = "gh",
        order: int = 3,
    ):
        # If didn't supply list, make a list
        if isinstance(keys, list):
            self.keys = keys
        else:
            self.keys = [keys]

        # Setup cubature method for factor
        if cubature == "gh":
            self._cubature_fun: Callable = gh_cubature

        elif cubature == "spherical":
            self._cubature_fun: Callable = spherical_cubature

        elif cubature == "unscented":
            self._cubature_fun: Callable = unscented_cubature

        elif cubature == "student_t":
            self._cubature_fun: Callable = student_t_cubature
        
        elif cubature == "trans_spherical":
            self._cubature_fun: Callable = trans_spherical_cubature
        
        elif cubature == "trans_gh":
            self._cubature_fun: Callable = trans_gh_cubature
        
        elif cubature == "trans_unscented":
            self._cubature_fun: Callable = trans_unscented_cubature
        
        else:
            valid_cubs = ['gh', 'spherical', 'unscented', 'student_t', 'trans_gh', 'trans_spherical', 'trans_unscented']
            raise ValueError(f"The field cubature must be in {valid_cubs}")
        
        self._order: int = order
        self.type: str = None
        # State Slices for information/covariance matrices
        self.state_slices: List[slice] = [variable_slices[k] for k in self.keys]

        # Projection Matrix
        self.projection: np.ndarray = projection.copy()

        self._total_dof: int = None
        self._dof: int = None

        self._unit_sigma_pts: np.ndarray = None
        self._weights: np.ndarray = None

        self.expect_scalar: np.ndarray = None
        self.expect_column: np.ndarray = None
        self.expect_matrix: np.ndarray = None

    def _gen_unit_sigma_pts(self) -> Tuple[np.ndarray, np.ndarray]:
        unit_sp, weights = self._cubature_fun(
            state_dof=self._total_dof, order_p=self._order
        )
        return unit_sp, weights

    def _phi_dx(self, column_expect: np.ndarray, information: np.ndarray):
        return information @ column_expect

    def _phi_dx_dx(
        self,
        expect_scalar: np.ndarray,
        expect_matrix: np.ndarray,
        information: np.ndarray,
    ):
        a = information @ expect_matrix @ information
        b = information * expect_scalar
        return a - b

    @abstractmethod
    def _gen_sigma_pts(self, states: List[State], factor_covar: np.ndarray):
        """
        Generate new sigma points based off current covar.
        """
        pass

    @abstractmethod
    def _eval_factor(self, sigma_points: List[State]):
        """Evaluate the factor at current state estimate."""
        pass

    @abstractmethod
    def evaluate_derivatives(
        self, states: List[State], covar_matrix: np.ndarray, info_matrix: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns phi (cost), phi_dx (col vector), and phi_dx_dx (information matrix) associated with specific factor.
        """
        pass

    @abstractmethod
    def evaluate_factor_cost(
        self, states: List[State], covar_matrix: np.ndarray, info_matrix: np.ndarray
    ) -> np.ndarray:
        """
        Returns new phi (cost) based upon an updated state list, covariance and inforamtion matrices.
        """


class PriorFactor(Factor):
    "Generic Prior Factor"

    def __init__(
        self,
        keys: List[Hashable],
        prior_state: State,
        prior_covariance: np.ndarray,
        variable_slices: Dict[str, slice],
        projection: np.ndarray,
        cubature: str = "gh",
        order: int = 3,
    ):
        super().__init__(keys, variable_slices, projection, cubature, order)

        self.type = "prior"

        # Setup factor specific values
        self._prior_covariance = prior_covariance.copy()
        self._inv_prior_covariance = scipy.linalg.inv(self._prior_covariance)
        self._x0 = prior_state.copy()

        # Factor size
        self._dof, self._total_dof = prior_state.dof, prior_state.dof

        # Handle Errors
        if self._prior_covariance.shape != (self._dof, self._dof):
            raise ValueError(
                f"Prior covariance must have shape ({prior_state.dof}, {prior_state.dof})."
            )

        if self.projection.shape[0] != self._dof:
            raise ValueError(f"Projection matrix must have {prior_state.dof} rows.")

        # Generate the unit sigma points
        self._unit_sigma_pts, self._weights = self._gen_unit_sigma_pts()

    def _gen_sigma_pts(self, states: List[State], factor_covar: np.ndarray):
        """
        Generates a list, of a list of states with values corresponding to sigma point values.
        """

        sqrt_covariance = np.linalg.cholesky(factor_covar)
        vector_sigma_points = [
            sqrt_covariance @ sp_i.reshape((-1, 1)) for sp_i in self._unit_sigma_pts
        ]
        x = states[0]
        sigma_pts: List[List[State]] = [
            [x.plus(sp_vec)] for sp_vec in vector_sigma_points
        ]
        return sigma_pts

    def _eval_factor(self, sigma_points: List[State]):
        prior_diff = sigma_points[0].minus(self._x0).reshape((-1, 1))
        phi_prior = 0.5 * prior_diff.T @ self._inv_prior_covariance @ prior_diff
        return phi_prior

    def evaluate_derivatives(
        self, states: List[State], covar_matrix: np.ndarray, info_matrix: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns phi (cost), phi_dx (col vector), and phi_dx_dx (information matrix) associated with specific factor.
        """
        # Get covariance/information at factor level
        factor_covar = covar_matrix[self.state_slices[0], self.state_slices[0]]
        # factor_info = info_matrix[self.state_slices[0], self.state_slices[0]]
        factor_info = force_sym(scipy.linalg.inv(factor_covar))
        # Calculate sigma points from this new covariance
        sigma_points = self._gen_sigma_pts(states, factor_covar)
        # Get current state
        x = states[0]

        # Scalar Valued
        expect_phi = np.zeros((1, 1), dtype=np.float64)
        # Column Valued
        expect_mu_phi = np.zeros((self._dof, 1), dtype=np.float64)
        # Matrix Valued
        expect_mu_mu_phi = np.zeros((self._dof, self._dof), dtype=np.float64)

        # TODO: Vectorize this
        for i, w in enumerate(self._weights):
            phi_k = self._eval_factor(sigma_points[i])
            expect_phi += w * phi_k
            diff = sigma_points[i][0].minus(x).reshape((-1, 1))
            expect_mu_phi += w * diff * phi_k
            expect_mu_mu_phi += w * (diff @ diff.T) * phi_k

        global_column = self.projection.T @ self._phi_dx(expect_mu_phi, factor_info)
        global_matrix = (
            self.projection.T
            @ self._phi_dx_dx(expect_phi, expect_mu_mu_phi, factor_info)
            @ self.projection
        )

        return expect_phi, global_column, global_matrix

    def evaluate_factor_cost(self, states, covar_matrix, info_matrix):
        # Get covariance/information at factor level
        factor_covar = covar_matrix[self.state_slices[0], self.state_slices[0]]
        # Calculate sigma points from this new covariance
        sigma_points = self._gen_sigma_pts(states, factor_covar)

        expect_phi = np.zeros((1, 1), dtype=np.float64)

        # TODO: Vectorize this
        for i, w in enumerate(self._weights):
            phi_k = self._eval_factor(sigma_points[i])
            expect_phi += w * phi_k

        return expect_phi


class MeasurementFactor(Factor):
    "Generic Measurement Factor (non-SLAM)."

    def __init__(
        self,
        keys: List[Hashable],
        measurement: Measurement,
        variable_slices: Dict[str, slice],
        projection: np.ndarray,
        cubature: str = "gh",
        order: int = 3,
    ):
        super().__init__(keys, variable_slices, projection, cubature, order)

        # Setup factor specific values
        self.type = "meas"
        self._meas_val: np.ndarray = measurement.value
        self._meas_model: nav.MeasurementModel = measurement.model

        # Factor dof
        self._dof = self.projection.shape[0]
        self._total_dof = self._dof

        # Generate the unit sigma points
        self._unit_sigma_pts, self._weights = self._gen_unit_sigma_pts()

    def _gen_sigma_pts(self, states: List[State], factor_covar: np.ndarray):
        """
        Generates a list, of a list of states with values corresponding to sigma point values.
        """

        sqrt_covariance = np.linalg.cholesky(factor_covar)
        vector_sigma_points = [
            sqrt_covariance @ sp_i.reshape((-1, 1)) for sp_i in self._unit_sigma_pts
        ]
        x = states[0]
        sigma_pts: List[List[State]] = [
            [x.plus(sp_vec)] for sp_vec in vector_sigma_points
        ]
        return sigma_pts

    def _eval_factor(self, sigma_points: List[State]):
        meas_diff = (
            self._meas_val - self._meas_model.evaluate(sigma_points[0])
        ).reshape((-1, 1))
        R_k = self._meas_model.covariance(sigma_points[0])
        R_k_inv = force_sym(
            scipy.linalg.inv(np.atleast_2d(R_k))
        )
        # Gaussian Loss
        phi_meas = 0.5 * meas_diff.T @ R_k_inv @ meas_diff
        # Cauchy Loss Measurement
        # phi_meas = 0.5 * (3 + 1) * np.log(1.0 + (meas_diff.T @ R_k_inv @ meas_diff / 3))
        # Skew-Laplace Loss
        # lam = 0.1
        # alpha = np.sqrt(1 + lam**2 / R_k[0,0])
        # phi_meas = (-1 * lam * meas_diff @ R_k_inv ) + (alpha * np.sqrt(R_k_inv) @ np.abs(meas_diff))
        return phi_meas

    def evaluate_derivatives(
        self, states: List[State], covar_matrix: np.ndarray, info_matrix: np.ndarray
    ):
        # Get covariance/information at factor level
        factor_covar = covar_matrix[self.state_slices[0], self.state_slices[0]]
        # factor_info = info_matrix[self.state_slices[0], self.state_slices[0]]
        factor_info = force_sym(scipy.linalg.inv(factor_covar))
        # Calculate sigma points from this factors covariance
        sigma_points = self._gen_sigma_pts(states, factor_covar)
        # Get current state
        x = states[0]

        expect_phi = np.zeros((1, 1))
        expect_mu_phi = np.zeros((self._dof, 1), dtype=np.float64)
        expect_mu_mu_phi = np.zeros((self._dof, self._dof), dtype=np.float64)

        # TODO: Vectorize this
        for i, w in enumerate(self._weights):
            phi_k = self._eval_factor(sigma_points[i])
            expect_phi += w * phi_k
            diff = sigma_points[i][0].minus(x).reshape((-1, 1))
            expect_mu_phi += w * diff * phi_k
            expect_mu_mu_phi += w * (diff @ diff.T) * phi_k

        global_column = self.projection.T @ self._phi_dx(expect_mu_phi, factor_info)
        global_matrix = (
            self.projection.T
            @ self._phi_dx_dx(expect_phi, expect_mu_mu_phi, factor_info)
            @ self.projection
        )

        return expect_phi, global_column, global_matrix

    def evaluate_factor_cost(self, states, covar_matrix, info_matrix):
        # Get covariance/information at factor level
        factor_covar = covar_matrix[self.state_slices[0], self.state_slices[0]]

        # Calculate sigma points from this new covariance
        sigma_points = self._gen_sigma_pts(states, factor_covar)

        expect_phi = np.zeros((1, 1), np.float64)

        # TODO: Vectorize this
        for i, w in enumerate(self._weights):
            phi_k = self._eval_factor(sigma_points[i])
            expect_phi += w * phi_k

        return expect_phi


class ProcessFactor(Factor):
    "General Process Factor"

    def __init__(
        self,
        keys: List[Hashable],
        process_model: nav.ProcessModel,
        input: nav.Input,
        variable_slices: Dict[str, slice],
        projection: np.ndarray,
        cubature: str = "gh",
        order: int = 3,
    ):
        super().__init__(keys, variable_slices, projection, cubature, order)

        if len(self.keys) != 2:
            raise ValueError("Process factor must depend on two states.")

        self.type = "proc"
        self._process_model = process_model
        self._input = input

        # Factor size
        self._total_dof = self.projection.shape[0]
        self._dof = int(self._total_dof / 2)

        # Generate the unit sigma points
        self._unit_sigma_pts, self._weights = self._gen_unit_sigma_pts()

    def _gen_sigma_pts(
        self, states: List[State], factor_covar: np.ndarray
    ) -> List[List[State]]:
        """
        Generates a list, of a list of states with values corresponding to sigma point values.
        Where the first value of element is the sigma point corresponding to previous state distribution, and
        second element corresponds to current state estimate's distribution.
        """
        if factor_covar.shape[0] != self._total_dof:
            raise ValueError(
                f"Provided factor covariance needs to be ({self._total_dof},{self._total_dof})"
            )
        try:
            sqrt_covariance = np.linalg.cholesky(factor_covar)
            # sqrt_covariance = scipy.linalg.sqrtm(factor_covar)
        except np.linalg.LinAlgError as e:
            print(factor_covar)
            raise np.linalg.LinAlgError(
                "Process Factor joint covariance not Positive-Definite"
            )
        # vector_sigma_points = [
        #     sqrt_covariance @ sp_i.reshape((-1, 1)) for sp_i in self._unit_sigma_pts
        # ]
        vector_sigma_points = []
        for sp_i in self._unit_sigma_pts:
            xi_i = sqrt_covariance @ sp_i.reshape((-1, 1))
            vector_sigma_points.append(xi_i)

        x_km1 = states[0]
        x_k = states[1]
        sigma_pts = []
        for sp_vec in vector_sigma_points:
            sp_vec_prev = sp_vec[0 : self._dof]
            sp_vec_cur = sp_vec[self._dof :]
            sp_lie_prev = x_km1.plus(sp_vec_prev)
            sp_lie_cur = x_k.plus(sp_vec_cur)
            sigma_pts.append([sp_lie_prev, sp_lie_cur])

        return sigma_pts

    def _eval_factor(self, sigma_points: List[State]):
        sp_km1 = sigma_points[0]
        sp_k = sigma_points[1]
        # TODO: Check stamps
        dt = sp_k.stamp - sp_km1.stamp
        assert dt > 0.0
        propagated = self._process_model.evaluate(sp_km1, self._input, dt)
        Q_k = force_sym_PSD(self._process_model.covariance(sp_km1, self._input, dt))
        Q_k_inv = scipy.linalg.inv(Q_k)
        # TODO: Check this
        if isinstance(propagated, MatrixLieGroupState):
            assert propagated.direction == sp_k.direction
        process_diff = sp_k.minus(propagated).reshape((-1, 1))
        phi_proc = 0.5 * process_diff.T @ Q_k_inv @ process_diff
        return phi_proc

    def evaluate_derivatives(
        self, states: List[State], covar_matrix: np.ndarray, info_matrix: np.ndarray
    ):
        # Get covariance/information at state level

        x_km1_covar = covar_matrix[self.state_slices[0], self.state_slices[0]]

        x_km1_info = info_matrix[self.state_slices[0], self.state_slices[0]]

        x_k_covar = covar_matrix[self.state_slices[1], self.state_slices[1]]
        x_k_info = info_matrix[self.state_slices[1], self.state_slices[1]]

        cross_covar = covar_matrix[self.state_slices[0], self.state_slices[1]]
        cross_info = info_matrix[self.state_slices[0], self.state_slices[1]]

        # Form factor level covariance and information
        factor_covar = np.block(
            [[x_km1_covar, cross_covar], [cross_covar.T, x_k_covar]]
        )
        factor_covar_proj = self.projection @ covar_matrix @ self.projection.T

        assert np.allclose(factor_covar, factor_covar_proj)
        assert isPD(
            factor_covar
        ), f"Process Factor Covar: \n {factor_covar}\nis not positive-definite"

        # factor_info = np.block([[x_km1_info, cross_info], [cross_info.T, x_k_info]])
        factor_info = force_sym(scipy.linalg.inv(factor_covar))
        # Calculate sigma points from this new covariance
        sigma_points = self._gen_sigma_pts(states, factor_covar)
        # Get current state
        x_km1 = states[0]
        x_k = states[1]

        expect_phi = np.zeros((1, 1))
        expect_mu_phi = np.zeros((self._total_dof, 1), dtype=np.float64)
        expect_mu_mu_phi = np.zeros(
            (self._total_dof, self._total_dof), dtype=np.float64
        )

        # TODO: Vectorize this
        for i, w in enumerate(self._weights):
            phi_k = self._eval_factor(sigma_points[i])
            expect_phi += w * phi_k
            prev_diff = sigma_points[i][0].minus(x_km1).reshape((-1, 1))
            cur_diff = sigma_points[i][1].minus(x_k).reshape((-1, 1))
            diff = np.vstack((prev_diff, cur_diff))
            expect_mu_phi += w * diff * phi_k
            expect_mu_mu_phi += w * (diff @ diff.T) * phi_k

        global_column = self.projection.T @ self._phi_dx(expect_mu_phi, factor_info)
        global_matrix = (
            self.projection.T
            @ self._phi_dx_dx(expect_phi, expect_mu_mu_phi, factor_info)
            @ self.projection
        )

        return expect_phi, global_column, global_matrix

    def evaluate_factor_cost(
        self, states: List[State], covar_matrix: np.ndarray, info_matrix: np.ndarray
    ):
        # Get covariance/information at state level
        x_km1_covar = covar_matrix[self.state_slices[0], self.state_slices[0]]

        x_k_covar = covar_matrix[self.state_slices[1], self.state_slices[1]]

        cross_covar = covar_matrix[self.state_slices[0], self.state_slices[1]]

        # Form factor level covariance and information
        factor_covar = np.block(
            [[x_km1_covar, cross_covar], [cross_covar.T, x_k_covar]]
        )
        factor_covar_proj = self.projection @ covar_matrix @ self.projection.T
        assert np.allclose(factor_covar, factor_covar_proj)
        assert isPD(
            factor_covar
        ), f"Process Factor Covar: \n {factor_covar}\nis not positive-definite"
        # Calculate sigma points from this new covariance
        sigma_points = self._gen_sigma_pts(states, factor_covar)

        expect_phi = np.zeros((1, 1), dtype=np.float64)

        # TODO: Vectorize this
        for i, w in enumerate(self._weights):
            phi_k = self._eval_factor(sigma_points[i])
            expect_phi += w * phi_k

        return expect_phi

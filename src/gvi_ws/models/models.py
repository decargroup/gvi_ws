# %%
import numpy as np
import navlie as nav
from scipy import integrate
from scipy import stats
from navlie.types import MeasurementModel, State, Measurement, ProcessModel
from navlie.lib.states import VectorState, VectorInput, SE2State


class WheelEncoder(MeasurementModel):
    def __init__(self, R_d: np.ndarray) -> None:
        # R_d = diag(fwd_vel_var, side_slip_var, ang_vel_var)
        self.R = np.atleast_2d(R_d)
        return

    def evaluate(self, x: nav.State):
        if isinstance(x, SE2State):
            theta = x.group.Log(x)[0]
        elif isinstance(x, VectorState):
            theta = x.value[0]
        else:
            raise ValueError("WheelEncoder must take SE2State or VectorState as input.")
        y = np.zeros((3, 1))
        y[0, 0] = np.cos(theta) + np.sin(theta)
        y[1, 0] = -np.sin(theta) + np.cos(theta)
        y[2, 0] = 1
        return y

    def covariance(self, x):
        return self.R


class LaserRangeFinder(MeasurementModel):
    def __init__(self, R_d) -> None:
        self.R = R_d

    def evaluate(self, x: nav.State) -> np.ndarray:
        return x.value[0]

    def covariance(self, x: nav.State) -> np.ndarray:
        return self.R


class NonLinearLaserRangeFinder(MeasurementModel):
    def __init__(self, R_d, height, distance) -> None:
        self.R = R_d
        self.height = height
        self.distance = distance

    def evaluate(self, x: nav.State) -> np.ndarray:
        if x.value[0] + self.distance <= 0:
            print(f"Mass goes beyond the wall, position: {x.value[0]}")
            raise ValueError()
        range_measure = np.sqrt(
            np.square(x.value[0] + self.distance) + np.square(self.height)
        )
        return range_measure

    def covariance(self, x: nav.State) -> np.ndarray:
        return np.atleast_2d(self.R)


class StereoCamera(MeasurementModel):
    def __init__(self, R_d: np.ndarray, landmark_pos: np.ndarray, f=400, b=0.1):
        self.R = np.atleast_2d(R_d)
        self.landmark_pos = np.atleast_1d(landmark_pos)
        self.focal_len = f
        self.disparity = b

    def evaluate(self, x: nav.State) -> np.ndarray:
        y = np.atleast_2d(
            self.focal_len * self.disparity / (self.landmark_pos - x.value[0])
        )
        return y

    def covariance(self, x: nav.State) -> np.ndarray:
        return self.R


class DoubleIntegrator(ProcessModel):
    """
    The double-integrator process model is a second-order point kinematic model
    given in continuous time by

    .. math::
        \dot{\mathbf{r}} = \mathbf{v}

        \dot{\mathbf{v}} = \mathbf{u}

    where :math:`\mathbf{u}` is the input.
    """

    def __init__(self, Q: np.ndarray):
        """
        Parameters
        ----------
        Q : np.ndarray
            Q: Discrete time covariance on the input u.
        """
        if Q.shape[0] != Q.shape[1]:
            raise ValueError("Q must be an n x n matrix.")

        self._Q = Q
        self.dim = Q.shape[0]

    def evaluate(self, x: VectorState, u: VectorInput, dt: float) -> State:
        x_new = x.copy()
        Ad = self.jacobian(None, None, dt)
        Ld = self.input_jacobian(dt)
        u = np.atleast_1d(u.value)
        x_new.value = (
            Ad @ x.value.reshape((-1, 1)) + Ld @ u[: self.dim].reshape((-1, 1))
        ).ravel()
        return x_new

    def jacobian(self, x, u, dt) -> np.ndarray:
        Ad = np.identity(2 * self.dim)
        Ad[0 : self.dim, self.dim :] = dt * np.identity(self.dim)
        return Ad

    def covariance(self, x, u, dt) -> np.ndarray:
        Ld = self.input_jacobian(dt)
        sigma_acc = np.sqrt(self._Q * dt)
        Q = np.array([[(1 / 3) * dt**3, 0.5 * dt**2], [0.5 * dt**2, dt]]) * sigma_acc**2
        # return Ld @ self._Q @ Ld.T
        return Q

    def input_covariance(self, x, u, dt):
        return self._Q

    def input_jacobian(self, dt):
        Ld = np.zeros((2 * self.dim, self.dim))
        Ld[0 : self.dim, :] = 0.5 * dt**2 * np.identity(self.dim)
        Ld[self.dim :, :] = dt * np.identity(self.dim)
        return Ld

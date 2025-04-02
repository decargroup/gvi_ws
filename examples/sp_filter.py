# %%
import numpy as np
import scipy as sp
import navlie as nav

import matplotlib.pyplot as plt
from src.models.models import Simulator, NonLinearLaserRangeFinder, LaserRangeFinder

# Navlie Things
from navlie.lib.states import VectorState, VectorInput
from navlie.lib.models import RangePointToAnchor
from navlie.types import ProcessModel, MeasurementModel
from navlie.utils import plot_poses, plot_meas, plot_nees
from navlie.filters import (
    ExtendedKalmanFilter,
    SigmaPointKalmanFilter,
    GaussHermiteKalmanFilter,
    UnscentedKalmanFilter,
    CubatureKalmanFilter,
)

from typing import List


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

    def evaluate(self, x: VectorState, u: VectorInput, dt: float) -> np.ndarray:
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


# class MassSpringDamper(ProcessModel):
#     def __init__(self, m, k, c, T, Q_d):
#         self.m = m
#         self.c = c
#         self.k = k
#         self.f = lambda t: 0
#         self.A = np.array([[1, T],
#                            [-T*self.k/self.m, 1-T*self.c/self.m]])
#         self.B = np.array([[0],
#                            [T/self.m]])
#         self.T = T
#         self.Q_d = Q_d
#         self.dim = Q_d.shape[0]

#     def set_force(self, f) -> None:
#         self.f = f

#     def calc_force(self, t):
#         return np.array([[self.f(t)]])

#     def evaluate(self, x:VectorState, u:VectorInput, dt:float):
#         x = x.copy()
#         self.A = self.A = np.array([[1, dt],
#                            [-dt*self.k/self.m, 1-dt*self.c/self.m]])
#         self.B = np.array([[0],
#                            [dt/self.m]])

#         x.value = self.A @ x.value + self.B @ u.value
#         self.dt = dt
#         return x

#     def covariance(self, x=None, u=None, dt=None):
#         sigma_acc = np.sqrt(self.Q_d * self.T)
#         Q = np.array([[(1/3) * self.T**3, 0.5 * self.T**2],
#                     [0.5*self.T**2, self.T]]) * sigma_acc**2
#         return Q

#     # def input_covariance(self, x, u, dt):
#     #     return self.Q_d


# %%
if __name__ == "__main__":

    NOISE_ON = True
    SIGMA_METHOD = "unscented"  #'cubature' #'gh' #'unscented'
    ORDER_P = 2
    LINEAR = False
    np.random.seed(2)

    # Plotting parameters
    plt.rc("text", usetex=True)
    plt.rc("font", family="serif", size=14)
    plt.rc("lines", linewidth=2)
    plt.rc("axes", grid=True)
    plt.rc("grid", linestyle="--")

    laser_range_freq = 10
    imu_freq = 100
    sigma_acc_continuous = 0.045
    dt = 1 / imu_freq
    Q_d = np.array([[sigma_acc_continuous**2 / dt]])
    R_k = np.array([0.01])

    proc_model = DoubleIntegrator(Q_d)

    if LINEAR:
        laser_range = LaserRangeFinder(R_d=R_k)
    else:
        laser_range = NonLinearLaserRangeFinder(R_d=R_k, height=2, distance=7)

    x0_val = [5, 0]
    Simulation = Simulator(t_end=10, freq=imu_freq, x0=x0_val)
    # Set Forcing Function
    # Forcing function f(t) = A sin(wt)
    f = lambda t: 1 * np.sin(2 * np.pi * t)
    Simulation.set_forcing_function(f)
    # Generating ground truth
    true_pos, true_vel, true_acc = Simulation.generate_ground_truth()
    t = Simulation.time

    fig, ax = plt.subplots(3, 1, sharex=True)
    ax[0].set_ylabel(r"$x(t)$ (m)")
    # ax[0].set_xlabel(r'$t$ [s]')
    ax[0].plot(t, true_pos)

    ax[1].set_ylabel(r"$\dot{x}(t)$ (m/s)")
    # ax[1].set_xlabel(r'$t$ [s]')
    ax[1].plot(t, true_vel)

    ax[2].set_ylabel(r"$\ddot{x}(t)$ (m/s)")
    ax[2].set_xlabel(r"$t$ (s)")
    ax[2].plot(t, true_acc)

    measured_pos, measured_acc, measured_time = (
        Simulation.generate_nonlinear_measurements(
            sigma_acc=sigma_acc_continuous,
            pos_freq=laser_range_freq,
            acc_freq=imu_freq,
            meas_model=laser_range,
        )
    )
    fig, ax = plt.subplots(2, 1, sharex=True)
    ax[0].set_ylabel(r"$y(t)$ (m)")
    # ax[0].set_xlabel(r'$t$ (s)')
    ax[0].scatter(measured_time, measured_pos, s=0.05, marker="x")

    ax[1].set_ylabel(r"$u^{acc}(t)$ ($\frac{m}{s^2}$)")
    ax[1].set_xlabel(r"$t$ (s)")
    ax[1].scatter(Simulation.time, measured_acc, s=0.05)

    # Get Navlie formatted data
    gt_data, input_data, meas_data = Simulation.get_nav_info()
    x0 = VectorState(value=np.array(x0_val), stamp=gt_data[0].stamp)

    # %%
    # Kalman Filter
    # Run Filter
    P0 = np.diag([0.8, 0.8])

    if NOISE_ON:
        x0 = x0.plus(nav.randvec(P0))

    x_ekf = nav.StateWithCovariance(x0, P0)
    x_spkf = nav.StateWithCovariance(x0, P0)
    spkf = SigmaPointKalmanFilter(process_model=proc_model, method=SIGMA_METHOD)
    ekf = ExtendedKalmanFilter(process_model=proc_model)
    meas_idx = 0
    y = meas_data[meas_idx]
    results_list_ekf = []
    results_list_spkf = []
    for k in range(len(input_data) - 1):
        u = input_data[k]

        # Fuse any measurements that have occurred.
        while y.stamp < input_data[k + 1].stamp and meas_idx < len(meas_data):
            x_ekf = ekf.correct(x_ekf, y, u)
            x_spkf = spkf.correct(x_spkf, y, u)

            # Load the next measurement
            meas_idx += 1
            if meas_idx < len(meas_data):
                y = meas_data[meas_idx]

        dt = input_data[k + 1].stamp - x_ekf.state.stamp
        x_ekf = ekf.predict(x_ekf, u, dt)
        x_spkf = spkf.predict(x_spkf, u, dt)

        results_list_ekf.append(nav.GaussianResult(x_ekf, gt_data[k + 1]))
        results_list_spkf.append(nav.GaussianResult(x_spkf, gt_data[k + 1]))

    # ##############################################################################
    # Post processing
    results_ekf = nav.GaussianResultList(results_list_ekf)
    results_spkf = nav.GaussianResultList(results_list_spkf)
    # %%
    fig, ax = plt.subplots(2, 1)
    fig.tight_layout()
    ax[0].plot(results_ekf.stamp, results_ekf.value[:, 0], label="EKF")
    ax[0].plot(
        results_spkf.stamp, results_spkf.value[:, 0], label=f"SPKF ({SIGMA_METHOD})"
    )
    ax[0].plot(results_ekf.stamp, results_ekf.value_true[:, 0], label="Ground truth")
    ax[0].set_xlabel("t (s)")
    ax[0].set_ylabel("x (m)")
    ax[1].plot(results_ekf.stamp, results_ekf.value[:, 1], label="EKF")
    ax[1].plot(
        results_spkf.stamp, results_spkf.value[:, 1], label=f"SPKF ({SIGMA_METHOD})"
    )
    ax[1].plot(results_ekf.stamp, results_ekf.value_true[:, 1], label="Ground truth")
    ax[1].set_xlabel("t (s)")
    ax[1].set_ylabel("v (m/s)")
    ax[0].legend()
    ax[1].legend()
    fig.savefig(
        f"/home/astirl/Documents/courses/notes/mech_642/figs/msd_track_{SIGMA_METHOD}.pdf"
    )

    # %%
    fig, axs = plt.subplots(2, 1)
    fig.tight_layout()
    # axs: List[plt.Axes] = axs
    for i in range(len(axs)):
        axs[i].fill_between(
            results_ekf.stamp,
            results_ekf.three_sigma[:, i],
            -results_ekf.three_sigma[:, i],
            alpha=0.5,
        )
        axs[i].plot(results_ekf.stamp, results_ekf.error[:, i], label="EKF")
        axs[i].plot(
            results_spkf.stamp, results_spkf.error[:, i], label=f"SPKF ({SIGMA_METHOD})"
        )
    axs[0].set_title("Estimation error")
    axs[1].set_xlabel("Time (s)")
    axs[0].legend()
    axs[0].set_ylabel("x error (m)")
    axs[1].set_ylabel(r"v error $(\frac{m}{s})$")
    axs[1].legend()
    fig.savefig(
        f"/home/astirl/Documents/courses/notes/mech_642/figs/msd_sigma_{SIGMA_METHOD}.pdf"
    )
    plt.show()

    # %%
    fig, ax = plt.subplots(2, 1)
    fig.tight_layout()
    ax[0].set_title("Filter Difference")
    ax[0].plot(results_ekf.stamp, results_ekf.value[:, 0] - results_spkf.value[:, 0])
    ax[0].set_xlabel("t (s)")
    ax[0].set_ylabel(r"$\delta$x (m)")
    ax[1].plot(results_ekf.stamp, results_ekf.value[:, 1] - results_spkf.value[:, 1])
    ax[1].set_xlabel("t (s)")
    ax[1].set_ylabel(r"$\delta$ v $(\frac{m}{s})$")
    ax[0].legend()
    ax[1].legend()
    fig.savefig(
        f"/home/astirl/Documents/courses/notes/mech_642/figs/msd_filterdiff_{SIGMA_METHOD}.pdf"
    )

    # %%
    fig, axs = plot_nees(
        results_ekf, confidence_interval=0.997, normalize=False, label="EKF"
    )
    fig, axs = plot_nees(
        results_spkf,
        confidence_interval=0.997,
        normalize=False,
        ax=axs,
        label=f"SPKF ({SIGMA_METHOD})",
    )
    fig.savefig(
        f"/home/astirl/Documents/courses/notes/mech_642/figs/msd_nees_{SIGMA_METHOD}.pdf"
    )
    # %%

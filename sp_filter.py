# %%
import numpy as np
import scipy as sp
import navlie as nav

import matplotlib.pyplot as plt

# Navlie Things
from navlie.datagen import DataGenerator
from navlie.lib.states import VectorState, VectorInput
from navlie.lib.models import RangePointToAnchor
from navlie.types import ProcessModel, MeasurementModel
from navlie.utils import plot_poses, plot_meas, plot_nees
from navlie.filters import ExtendedKalmanFilter, SigmaPointKalmanFilter, GaussHermiteKalmanFilter, UnscentedKalmanFilter, CubatureKalmanFilter

from typing import List

class MassSpringDamper(ProcessModel):
    def __init__(self, m, k, c, T, Q_d):
        self.m = m
        self.c = c
        self.k = k
        self.f = lambda t: 0
        self.A = np.array([[1, T],
                           [-T*self.k/self.m, 1-T*self.c/self.m]])
        self.B = np.array([[0],
                           [T/self.m]])
        self.Q_d = Q_d
        
    def set_force(self, f) -> None:
        self.f = f
    
    def calc_force(self, t):
        return np.array([[self.f(t)]])
    
    def evaluate(self, x:VectorState, u:VectorInput, dt:float):
        x = x.copy()
        self.A = self.A = np.array([[1, dt],
                           [-dt*self.k/self.m, 1-dt*self.c/self.m]])
        self.B = np.array([[0],
                           [dt/self.m]])
        
        x.value = self.A @ x.value + self.B @ u.value
        return x
    
    def covariance(self, x, u, dt):
        return self.B @ self.Q_d @ self.B.T
    


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
        range_measure = np.sqrt(np.square(x.value[0] + self.distance) + np.square(self.height))
        return range_measure
    
    def covariance(self, x: nav.State) -> np.ndarray:
        return self.R

# %%
NOISE_ON = True
SIGMA_METHOD = 'unscented' #'cubature' #'gh' #'unscented' 
ORDER_P = 2
LINEAR = False
np.random.seed(2)

# Plotting parameters
plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=14)
plt.rc('lines', linewidth=2)
plt.rc('axes', grid=True)
plt.rc('grid', linestyle='--')

Q_d = np.array([[0.1]])

msd = MassSpringDamper(1, 1, 1, 0.01, Q_d)

input_profile = lambda t, x: np.array([np.sin(t)])

R_d = np.array([0.01]) #m
if LINEAR:
    laser_range = LaserRangeFinder(R_d)
else:
    laser_range = NonLinearLaserRangeFinder(R_d, height=7, distance=3)
dg = DataGenerator(process_model=msd, input_func=input_profile, input_covariance=Q_d, input_freq=100, meas_model_list=[laser_range], meas_freq_list=10)


# %%
x0 = x0 = VectorState(np.array([1, 0]), stamp=0)
gt_data, input_data, meas_data = dg.generate(x0, 0, 10, noise=NOISE_ON)

gt_pos = []
gt_vel = []
for gt in gt_data:
    gt_pos.append([gt.stamp, gt.value[0]])
    gt_vel.append([gt.stamp, gt.value[1]])

input_data_vals = []
for input in input_data:
    input_data_vals.append([input.stamp, input.value])

meas_vals = []
for meas in meas_data:
    meas_vals.append([meas.stamp, meas.value])
meas_vals = np.array(meas_vals)
gt_pos = np.array(gt_pos)
gt_vel = np.array(gt_vel)

# %%
fig, ax = plt.subplots(3,1)
ax[0].plot(gt_pos[:,0],gt_pos[:,1] )
ax[0].set_title("True Position")
ax[1].plot(gt_vel[:,0],gt_vel[:,1] )
ax[1].set_title("True Velocity")
ax[2].plot(meas_vals[:,0],meas_vals[:,1] )
ax[2].set_title("Measurements")
fig.tight_layout()
fig.savefig(f'/home/astirl/Documents/courses/notes/mech_642/figs/msd_data.pdf')
# %%
# Kalman Filter
# Run Filter
P0 = np.diag([0.8, 0.8])

if NOISE_ON:
    x0 = x0.plus(nav.randvec(P0))

x_ekf = nav.StateWithCovariance(x0, P0)
x_spkf = nav.StateWithCovariance(x0, P0)
spkf = SigmaPointKalmanFilter(process_model=msd, method=SIGMA_METHOD)
ekf = ExtendedKalmanFilter(process_model=msd)
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
ax[0].plot(results_spkf.stamp, results_spkf.value[:, 0], label=f"SPKF ({SIGMA_METHOD})")
ax[0].plot(results_ekf.stamp, results_ekf.value_true[:, 0], label="Ground truth")
ax[0].set_xlabel("t (s)")
ax[0].set_ylabel("x (m)")
ax[1].plot(results_ekf.stamp, results_ekf.value[:, 1], label="EKF")
ax[1].plot(results_spkf.stamp, results_spkf.value[:, 1], label=f"SPKF ({SIGMA_METHOD})")
ax[1].plot(results_ekf.stamp, results_ekf.value_true[:, 1], label="Ground truth")
ax[1].set_xlabel("t (s)")
ax[1].set_ylabel("v (m/s)")
ax[0].legend()
ax[1].legend()
fig.savefig(f'/home/astirl/Documents/courses/notes/mech_642/figs/msd_track_{SIGMA_METHOD}.pdf')

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
    axs[i].plot(results_ekf.stamp, results_ekf.error[:, i], label='EKF')
    axs[i].plot(results_spkf.stamp,results_spkf.error[:, i], label=f'SPKF ({SIGMA_METHOD})')
axs[0].set_title("Estimation error")
axs[1].set_xlabel("Time (s)")
axs[0].legend()
axs[0].set_ylabel("x error (m)")
axs[1].set_ylabel(r"v error $(\frac{m}{s})$")
axs[1].legend()
fig.savefig(f'/home/astirl/Documents/courses/notes/mech_642/figs/msd_sigma_{SIGMA_METHOD}.pdf')
plt.show()

# %%
fig, ax = plt.subplots(2, 1)
fig.tight_layout()
ax[0].set_title("Filter Difference")
ax[0].plot(results_ekf.stamp, results_ekf.value[:, 0] - results_spkf.value[:,0])
ax[0].set_xlabel("t (s)")
ax[0].set_ylabel(r"$\delta$x (m)")
ax[1].plot(results_ekf.stamp, results_ekf.value[:, 1] - results_spkf.value[:,1])
ax[1].set_xlabel("t (s)")
ax[1].set_ylabel(r"$\delta$ v $(\frac{m}{s})$")
ax[0].legend()
ax[1].legend()
fig.savefig(f'/home/astirl/Documents/courses/notes/mech_642/figs/msd_filterdiff_{SIGMA_METHOD}.pdf')

# %%
fig, axs = plot_nees(results_ekf, confidence_interval=0.997, normalize=False, label='EKF')
fig, axs = plot_nees(results_spkf, confidence_interval=0.997, normalize=False, ax=axs, label=f'SPKF ({SIGMA_METHOD})')
fig.savefig(f'/home/astirl/Documents/courses/notes/mech_642/figs/msd_nees_{SIGMA_METHOD}.pdf')
# %%

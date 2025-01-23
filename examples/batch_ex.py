# %%
import navlie as nav 
import numpy as np
import typing
import matplotlib.pyplot as plt
from examples.gh_quad_ex import gh_cubature_nav, gh_cubature
from examples.sp_filter import MassSpringDamper, NonLinearLaserRangeFinder, LaserRangeFinder
from navlie.datagen import DataGenerator
from navlie.lib.states import VectorState, VectorInput
from navlie.lib.models import RangePointToAnchor
from navlie.types import ProcessModel, MeasurementModel, Measurement, Input, StateWithCovariance, State
from navlie.utils import find_nearest_stamp_idx


NOISE_ON = True
ORDER_P = 2
Q_d = np.array([[0.1]])
R_d = np.array([0.05]) #m
msd = MassSpringDamper(m=1, k=1, c=1, T=0.01, Q_d=Q_d)
input_profile = lambda t, x: np.array([np.sin(t)])
laser_range = NonLinearLaserRangeFinder(R_d, height=7, distance=3)
dg = DataGenerator(process_model=msd, input_func=input_profile, input_covariance=Q_d, input_freq=100, meas_model_list=[laser_range], meas_freq_list=10)
x0 =  VectorState(np.array([1, 0]), stamp=0)
P0 = np.diag([0.8, 0.8])
gt_data, input_data, meas_data = dg.generate(x0, 0, 5, noise=NOISE_ON)
x0 = StateWithCovariance(x0, P0)
gt_data = gt_data[0:2]
input_data = input_data[0:2]
meas_data = meas_data[0:1]


estimator = nav.BatchEstimator(solver_type="LM", max_iters=20, step_tol=None, gradient_tol=1e-7, ftol=1e-8, verbose=True)
estimate_list, opt_results = estimator.solve(x0=x0.state, P0 = x0.covariance, input_data=input_data, process_model=msd, meas_data=meas_data, return_opt_results=True)

estimate_stamps = [float(x.state.stamp) for x in estimate_list]
gt_stamps = [x.stamp for x in gt_data]

matches = nav.associate_stamps(estimate_stamps, gt_stamps)

est_list = []
gt_list = []
for match in matches:
    gt_list.append(gt_data[match[1]])
    est_list.append(estimate_list[match[0]])

# Postprocess the results and plot
results = nav.GaussianResultList.from_estimates(est_list, gt_list)

fig, ax = nav.plot_error(results)
ax[0].set_title("Position")
ax[1].set_title("Velocity")
ax[0].set_xlabel("Time (s)")
ax[1].set_xlabel("Time (s)")
plt.tight_layout()
plt.show()

fig, ax = plt.subplots(2, 1)
fig.tight_layout()
ax[0].plot(results.stamp, results.value[:, 0], label="Batch")
ax[0].plot(results.stamp, results.value_true[:, 0], label="Ground truth")
ax[0].set_xlabel("t (s)")
ax[0].set_ylabel("x (m)")
ax[1].plot(results.stamp, results.value[:, 1], label="Batch")
ax[1].plot(results.stamp, results.value_true[:, 1], label="Ground truth")
ax[1].set_xlabel("t (s)")
ax[1].set_ylabel("v (m/s)")
ax[0].legend()
ax[1].legend()
plt.show()

# %%
import numpy as np
import navlie as nav
from navlie.utils import plot_poses, plot_meas, plot_nees
from gvi import GVI
from models import Simulator, NonLinearLaserRangeFinder, LaserRangeFinder, DoubleIntegrator, StereoCamera
from factors import construct_factor_list
from navlie.lib.states import VectorState
from navlie.types import  StateWithCovariance
from navlie.filters import ExtendedKalmanFilter, IteratedKalmanFilter
from util.psd import force_PSD
import matplotlib.pyplot as plt
from scipy.linalg import block_diag
from abc import abstractmethod
import timeit

if __name__== '__main__':
    NOISE_ON = True
    LINEAR = False
    STEREO = True
    BACKTRACK = False
    TIME_IT = True
    SIM_TIME = 10
    ITERATE = False
    # MAX_LEN = 40
    np.random.seed(1)

    # Plotting parameters
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif', size=14)
    plt.rc('lines', linewidth=2)
    plt.rc('axes', grid=True)
    plt.rc('grid', linestyle='--')

    ######## SIM SETUP ###########
    laser_range_freq = 10
    imu_freq = 100
    sigma_acc_continuous = 0.5
    dt = 1 / imu_freq
    R_k = np.array([0.05])
    if LINEAR:
        meas_model = LaserRangeFinder(R_d=R_k)
    else:
        if STEREO:
            R_k = np.array([0.01])
            landmark_pos = np.array([10])
            meas_model = StereoCamera(R_d=R_k, landmark_pos=landmark_pos)
        else:
            meas_model = NonLinearLaserRangeFinder(R_d=R_k, height=5, distance=8)

    x0_val = [5, 0]
    Simulation = Simulator(t_end=SIM_TIME, freq=imu_freq, x0=x0_val)
    # Set Forcing Function
    # Forcing function f(t) = A sin(wt)
    f = lambda t: 1 * np.sin(2*np.pi*t)
    Simulation.set_forcing_function(f)
    # Generating ground truth
    true_pos, true_vel, true_acc = Simulation.generate_ground_truth()

    meas_pos,_,meas_t = Simulation.generate_measurements(sigma_acc=sigma_acc_continuous, pos_freq=laser_range_freq, acc_freq=imu_freq, meas_model=meas_model, add_noise=NOISE_ON)

    ####################
    #### IEKF Setup #####
    ####################
    # Get Navlie formatted data
    gt_data, input_data, meas_data = Simulation.get_nav_info()
    input_data_lim = input_data[:]
    meas_data_lim = meas_data[:]
    gt_data_lim = gt_data[:]
    # input_data_lim = input_data[0:2]
    # meas_data_lim = meas_data[0:1]
    
    state_dof = len(x0_val)
    x0_state = VectorState(value=np.array(x0_val), stamp=gt_data[0].stamp)
    # sigma_acc_continuous = 100
    dt = 1 / imu_freq
    Q_d = np.array([[sigma_acc_continuous**2 / dt]])
    proc_model = DoubleIntegrator(Q_d)
    P0 = np.eye(2) * 1e-2
    if NOISE_ON:
        x0_state = x0_state.plus(nav.randvec(P0))
    x0 = StateWithCovariance(state=x0_state.copy(), covariance=P0)


    ekf = ExtendedKalmanFilter(process_model=proc_model)
    iekf = IteratedKalmanFilter(process_model=proc_model)

    meas_idx = 0
    y = meas_data[meas_idx]
    ekf_results_list = []
    iekf_results_list = []
    x_ekf = x0.copy()
    x_ekf.state.value = np.copy(x0.state.value)
    x_iekf = x0.copy()
    x_iekf.state.value = np.copy(x0.state.value)
    for k in range(len(input_data) - 1):
        u = input_data[k]

        # Fuse any measurements that have occurred.
        while y.stamp < input_data[k + 1].stamp and meas_idx < len(meas_data):
            x_ekf = ekf.correct(x_ekf, y, u)
            x_iekf = iekf.correct(x_iekf, y, u)
            # Load the next measurement
            meas_idx += 1
            if meas_idx < len(meas_data):
                y = meas_data[meas_idx]

        dt = input_data[k + 1].stamp - x_ekf.state.stamp
        x_ekf = ekf.predict(x_ekf, u, dt)
        x_iekf = iekf.predict(x_iekf, u, dt)

        ekf_results_list.append(nav.GaussianResult(x_ekf, gt_data[k + 1]))
        iekf_results_list.append(nav.GaussianResult(x_iekf, gt_data[k + 1]))


    # Post processing
    results_ekf = nav.GaussianResultList(ekf_results_list)
    results_iekf = nav.GaussianResultList(iekf_results_list)

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
        axs[i].plot(results_iekf.stamp,results_iekf.error[:, i], label=f'IEKF')
    axs[0].set_title("Estimation error")
    axs[1].set_xlabel("Time (s)")
    axs[0].legend()
    axs[0].set_ylabel(r"$x \; (m)$")
    axs[1].set_ylabel(r"$\dot{x}\; (m/s)$")
    axs[1].legend()
    plt.show()

    fig, axs = plot_nees(results_ekf, confidence_interval=0.997, normalize=False, label='EKF')
    fig, axs = plot_nees(results_iekf, confidence_interval=0.997, normalize=False, ax=axs, label=f'EKF')


    


# %%

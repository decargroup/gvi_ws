import numpy as np
import navlie as nav
from gvi import GVI
from models import Simulator, NonLinearLaserRangeFinder, LaserRangeFinder, DoubleIntegrator
from factors import construct_factor_list
from navlie.lib.states import VectorState
from navlie.types import  StateWithCovariance
from util.psd import force_PSD
import matplotlib.pyplot as plt
from scipy.linalg import block_diag
from abc import abstractmethod


if __name__== '__main__':
    NOISE_ON = True
    LINEAR = False
    SIM_TIME = 0.5
    CUB_METHOD = 'GH'
    GH_DEG = 3
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
    sigma_acc_continuous = 0.045
    dt = 1 / imu_freq
    R_k = np.array([0.1])
    if LINEAR:
        laser_range = LaserRangeFinder(R_d=R_k)
    else:
        laser_range = NonLinearLaserRangeFinder(R_d=R_k, height=2, distance=8)

    x0_val = [5, 0]
    Simulation = Simulator(t_end=SIM_TIME, freq=imu_freq, x0=x0_val)
    # Set Forcing Function
    # Forcing function f(t) = A sin(wt)
    f = lambda t: 1 * np.sin(2*np.pi*t)
    Simulation.set_forcing_function(f)
    # Generating ground truth
    true_pos, true_vel, true_acc = Simulation.generate_ground_truth()

    _,_,_ = Simulation.generate_measurements(sigma_acc=sigma_acc_continuous, pos_freq=laser_range_freq, acc_freq=imu_freq, meas_model=laser_range, add_noise=NOISE_ON)
    ####################
    #### GVI SETUP #####
    ####################
    # Get Navlie formatted data
    gt_data, input_data, meas_data = Simulation.get_nav_info()
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

    factored_state_list = construct_factor_list(x0, input_data, meas_data, proc_model, cubature_type=CUB_METHOD, gh_deg=GH_DEG)

    ####################
    ##### RUN GVI ######
    ####################

    gvi = GVI(factored_states=factored_state_list, total_dim=state_dof*len(gt_data), debug=True)
    gvi.solve()

    #####################
    ##### PLOT GVI ######
    #####################
    estimate_list_gvi = gvi.get_estimate_list()
    estimate_stamps = [float(x.stamp) for x in estimate_list_gvi]
    gt_stamps = [x.stamp for x in gt_data]

    matches = nav.associate_stamps(estimate_stamps, gt_stamps)

    est_list_gvi = []
    gt_list = []
    for match in matches:
        gt_list.append(gt_data[match[1]])
        est_list_gvi.append(estimate_list_gvi[match[0]])

    results_gvi = nav.GaussianResultList.from_estimates(est_list_gvi, gt_list)
    fig_gvi, ax_gvi = nav.plot_error(results_gvi)
    ax_gvi[0].set_title("Position")
    ax_gvi[0].set_ylabel(r'$r$ (m)')
    ax_gvi[1].set_title("Velocity")
    ax_gvi[0].set_xlabel("Time (s)")
    ax_gvi[1].set_ylabel(r'$\dot{r}$ (m/s)')
    ax_gvi[1].set_xlabel("Time (s)")
    plt.tight_layout()
    plt.savefig('/home/astirl/Documents/courses/assignments/mech_642/gvi_ws/figs/esgvi_three_sig.pdf')

    # fig, ax = plt.subplots(2, 1)
    # fig.tight_layout()
    # ax[0].plot(results_gvi.stamp, results_gvi.value[:, 0], label="ESGVI")
    # ax[0].plot(results_gvi.stamp, results_gvi.value_true[:, 0], label="Ground truth")
    # ax[0].set_xlabel("t (s)")
    # ax[0].set_ylabel("x (m)")
    # ax[1].plot(results_gvi.stamp, results_gvi.value[:, 1], label="ESGVI")
    # ax[1].plot(results_gvi.stamp, results_gvi.value_true[:, 1], label="Ground truth")
    # ax[1].set_xlabel("t (s)")
    # ax[1].set_ylabel("v (m/s)")
    # ax[0].legend()
    # ax[1].legend()
    

    #######################
    ##### BATCH COMP ######
    #######################

    estimator = nav.BatchEstimator(solver_type="GN", verbose=True)
    estimate_list_map, opt_results = estimator.solve(x0=x0.state, P0 = x0.covariance, input_data=input_data, process_model=proc_model, meas_data=meas_data, return_opt_results=True)

    estimate_stamps_map = [float(x.state.stamp) for x in estimate_list_map]
    gt_stamps = [x.stamp for x in gt_data]

    matches = nav.associate_stamps(estimate_stamps_map, gt_stamps)

    est_list_map = []
    gt_list = []
    for match in matches:
        gt_list.append(gt_data[match[1]])
        est_list_map.append(estimate_list_map[match[0]])

    # Postprocess the results and plot
    results_map = nav.GaussianResultList.from_estimates(est_list_map, gt_list)

    fig, ax = nav.plot_error(results_map, label='MAP')
    ax[0].set_title("Position Error")
    ax[0].set_ylabel(r'$r$ (m)')
    ax[0].plot(results_gvi.stamp, results_gvi.error[:, 0], label='ESGVI', linestyle='--')
    ax[0].fill_between(results_gvi.stamp, results_gvi.three_sigma[:, 0], -results_gvi.three_sigma[:, 0], alpha=0.1, color='orange')
    ax[1].set_title("Velocity Error")
    ax[1].plot(results_gvi.stamp, results_gvi.error[:, 1], label='ESGVI', linestyle='--')
    ax[1].fill_between(results_gvi.stamp, results_gvi.three_sigma[:, 1], -results_gvi.three_sigma[:, 1], alpha=0.1, color='orange')
    ax[1].set_xlabel("Time (s)")
    ax[1].set_ylabel(r'$\dot{r}$ (m/s)')
    ax[0].legend(loc='upper right')
    ax[1].legend()
    plt.tight_layout()
    plt.savefig('/home/astirl/Documents/courses/assignments/mech_642/gvi_ws/figs/esgvi_map_three_sig.pdf')

    ##########################
    ##### PRINT RESULTS ######
    ##########################

    print(" Method |    Mean Pos Error     |  Mean Vel Error")
    print("-----------------------------------------------  ")
    print(f" ESGVI  | {np.mean(results_gvi.error[:,0])}  | {np.mean(results_gvi.error[:,1])}")
    print(f" MAP    | {np.mean(results_map.error[:,0])} | {np.mean(results_map.error[:,1])}")

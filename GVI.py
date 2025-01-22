# %%
import numpy as np
import scipy.linalg
import navlie as nav
from typing import Callable, Optional, List
from gh_quad_ex import gh_cubature_nav, gh_cubature
from sp_filter import DoubleIntegrator
from msd import Simulator, NonLinearLaserRangeFinder, LaserRangeFinder
from navlie.datagen import DataGenerator
from navlie.lib.states import VectorState, VectorInput
from navlie.lib.models import RangePointToAnchor
from navlie.types import ProcessModel, MeasurementModel, Measurement, Input, StateWithCovariance, State
from psd_util import force_PSD
from navlie.utils import find_nearest_stamp_idx
import matplotlib.pyplot as plt
from scipy.linalg import block_diag
from abc import abstractmethod

class FactoredState:
    def __init__(self, mean:np.ndarray, covariance:np.ndarray, proj_matrix:np.ndarray, stamp:float, cubature_type= 'GH', gh_degree = 3):
        self.mean:np.ndarray = mean.reshape((-1,1))
        self.covariance:np.ndarray = covariance.copy()
        self.sqrt_covariance:np.ndarray = np.linalg.cholesky(covariance)
        self.information:np.ndarray = force_PSD(scipy.linalg.inv(covariance))
        self.projection = proj_matrix.copy()
        self.stamp = stamp
        self.dof = np.shape(self.mean)[0]
        self.state_dof = self.dof
        
        # Cubature Method
        if cubature_type=='GH':
            self._gh_degree = gh_degree
            self.cubature:Callable = gh_cubature
        else:
            raise NotImplementedError("Implement other cubature methods")
        
        # Generate Unit Sigma Points
        self._unit_sigma_pts, self._weights = self.cubature(order_p=self._gh_degree, state_dof=self.dof)
        self.generate_new_sigma_pts()

    def generate_new_sigma_pts(self):
        self._sqrt_cov = np.linalg.cholesky(self.covariance)
        self._sigma_pts = [self.mean + self._sqrt_cov @ sp_i.reshape((-1,1)) 
                for sp_i in self._unit_sigma_pts]
        return
    
    def phi_dx(self):
        return self.information @ self._expect_mu_phi()

    def phi_dx_dx(self):
        a = self.information @ self._expect_mu_mu_phi() @ self.information
        b = self.information * self.expect_phi()
        return a - b
    
    def phi_dinfo(self):
        a = -0.5 * self._expect_mu_mu_phi()
        b = 0.5 * self.covariance * self.expect_phi()
        c = 0.5 * self.covariance
        return a + b + c

    def _expect_mu_mu_phi(self):
        expect = np.zeros_like(self.information)
        for i, w in enumerate(self._weights):
            expect += w * (self._sigma_pts[i] - self.mean) @ (self._sigma_pts[i] - self.mean).T  * self.eval_phi(self._sigma_pts[i])
        return expect

    def _expect_mu_phi(self):
        expect = np.zeros_like(self.mean)
        for i, w in enumerate(self._weights):
            phi = self.eval_phi(self._sigma_pts[i])
            expect += w * (self._sigma_pts[i] - self.mean) * phi
        return expect
    def expect_phi(self):
        # Scalar
        expect = 0
        for i, w in enumerate(self._weights):
            expect += w * self.eval_phi(self._sigma_pts[i]) 
        return expect
    
    def update_factor(self, total_mean, total_information):
        #TODO: Implement sparsity rules for info matrix
        total_covar = force_PSD(scipy.linalg.inv(total_information))
        self.mean = self.projection @ total_mean
        self.information = self.projection @ total_information @ self.projection.T
        self.covariance = self.projection @ total_covar @ self.projection.T
    
    # To be implemented by specific type of factors
    @abstractmethod
    def eval_phi(self, sigma_point:np.ndarray) -> np.ndarray:
        pass

    # Get functions
    def get_mean(self):
        return self.mean.copy()
    
    def get_information(self):
        return self.information.copy()
    
    def get_covariance(self):
        return self.covariance.copy()
            
class ProcessFactor(FactoredState):
    def __init__(self, mean, covariance, proj_matrix, stamp, cubature_type='GH', gh_degree=3):
        super().__init__(mean, covariance, proj_matrix, stamp, cubature_type, gh_degree)
        # Individual state dof
        self.state_dof = int(self.dof / 2)

    def link_dependent_state(self, process_model:ProcessModel, u_k_1:Input):
        self.process_model = process_model
        # TODO: Check if need this
        # self.prev_state = prev_state
        self.u = u_k_1
        self.dt = self.stamp - self.u.stamp
        # TODO: Fix the None arguments
        self.Q = process_model.covariance(None, None, self.dt)
        self.Q_inv = scipy.linalg.inv(self.Q)

    def eval_phi(self, sigma_point:np.ndarray):
        sp_x_k_1 = sigma_point.reshape((-1,1))[0:self.state_dof]
        sp_x_k = sigma_point.reshape((-1,1))[self.state_dof:]
        x_k_1 = VectorState(value=sp_x_k_1, stamp=self.u.stamp)
        proc_model_val = self.process_model.evaluate(x_k_1, self.u, self.dt).value.reshape((-1,1))
        phi = 0.5 * (sp_x_k - proc_model_val).T @ self.Q_inv @ (sp_x_k - proc_model_val)
        return phi
    
    def get_mean(self):
        return self.mean[self.state_dof:]
    
    def get_information(self):
        return self.information[self.state_dof:, self.state_dof:]
    
    def get_covariance(self):
        return self.covariance[self.state_dof:, self.state_dof:]

class PriorFactor(FactoredState):
    def __init__(self, mean, covariance, proj_matrix, stamp, cubature_type='GH', gh_degree=3):
        super().__init__(mean, covariance, proj_matrix, stamp, cubature_type, gh_degree)
        # Individual state dof
        self.state_dof = self.dof
    
    def link_prior(self, x0:StateWithCovariance):
        self.x0_check = x0.state.value.copy().reshape((-1,1))
        self.P0_check = x0.covariance.copy()
        self.P0_check_inv = scipy.linalg.inv(self.P0_check)

    def eval_phi(self, sigma_point:np.ndarray):
        phi = 0.5 * (sigma_point.reshape((-1,1)) - self.x0_check).T @ self.P0_check_inv @ (sigma_point.reshape((-1,1)) - self.x0_check)
        return phi
    
    def get_mean(self):
        return super().get_mean()
    def get_information(self):
        return super().get_information()
    def get_covariance(self):
        return super().get_covariance()

class MeasurementFactor(FactoredState):
    def __init__(self, mean, covariance, proj_matrix, stamp, cubature_type='GH', gh_degree=3):
        super().__init__(mean, covariance, proj_matrix, stamp, cubature_type, gh_degree)
        # Individual state dof
        self.state_dof = self.dof
    
    def link_measurement(self, meas:Measurement):
        self.y_k = meas.value.reshape((-1,1))
        self.meas_model = meas.model
        # TODO: Fix None argument
        self.R_k = np.atleast_2d(meas.model.covariance(x=None))
        self.R_k_inv = scipy.linalg.inv(self.R_k)

    def eval_phi(self, sigma_point:np.ndarray):
        sp_state = VectorState(value=sigma_point, stamp=self.stamp)
        phi = 0.5 * (self.y_k - self.meas_model.evaluate(sp_state).reshape((-1,1))).T @ self.R_k_inv @ (self.y_k - self.meas_model.evaluate(sp_state).reshape((-1,1)))
        return phi
    
    def get_mean(self):
        return super().get_mean()
    def get_information(self):
        return super().get_information()
    def get_covariance(self):
        return super().get_covariance()
    

class GVI:
    def __init__(self, factored_states:List[FactoredState], total_dim:int, debug=False):
        self.factored_states = factored_states
        self.total_dim = total_dim
        self.debug = debug
        # Initialize mean and information
        self.mean = np.zeros((total_dim, 1))
        self.information = np.zeros((total_dim, total_dim))
        k = 0
        for x_k in factored_states:
            # TODO: Check if there's a better way for initialization
            if isinstance(x_k, ProcessFactor):
                if k == 0:
                    dof = x_k.dof
                    state_dof = x_k.state_dof
                    self.mean[k:k+dof] = x_k.mean
                    self.information[k:k+dof, k:k+dof] = x_k.information
                    k+=dof
                else:
                    state_dof = x_k.state_dof
                    self.mean[k:k+state_dof] = x_k.get_mean()
                    self.information[k:k+state_dof, k:k+state_dof] = x_k.get_information()
                    # Cross Information Terms
                    self.information[k-state_dof:k, k:k+state_dof] = x_k.information[0:state_dof,state_dof:]
                    self.information[k:k+state_dof, k-state_dof:k] = x_k.information[state_dof:, 0:state_dof]
                    k+=state_dof
        self.information = force_PSD(self.information)
                
    
    def solve(self):
        n_iters = 0
        prev_cost = np.inf
        self.cur_cost = np.inf
        if self.debug:
            print("Starting Conditions: ")
            print("mu_{0}: \n", self.mean.T)
            print("Info_{0}: \n", self.information)
        while(True):
            # TODO: Implement backtracking
            # Update info
            phi_dx = np.zeros_like(self.mean)
            phi = np.zeros((1,1))
            new_information = np.zeros_like(self.information)
            functional_dinfo = np.zeros_like(self.information)
            for x_k in self.factored_states:
                # factored_state_k.generate_new_sigma_pts()
                proj_k = x_k.projection
                phi_dx += proj_k.T @ x_k.phi_dx()
                new_information += proj_k.T @ force_PSD(x_k.phi_dx_dx()) @ proj_k
                functional_dinfo += proj_k.T @ x_k.phi_dinfo() @ proj_k
                phi += x_k.expect_phi()
            
            
            self.cur_cost = phi + 0.5 * np.linalg.slogdet(self.information)[1]
            new_information = force_PSD(new_information)         
            delta_mu = np.linalg.solve(new_information, -phi_dx)
            delta_info = -2 * self.information @ functional_dinfo @ self.information
        
            # Calculate breaking condition
            size_mu = np.abs(np.linalg.norm(delta_mu))
            size_info = np.linalg.trace(delta_info)

            if size_mu < 1e-8 and size_info < 1e-8:
                print("--------------------------------")
                print(f"|  Converged in {n_iters} iterations!  |")
                print("--------------------------------")

                break
            if n_iters > 1000:
                print(f"Reached max iterations")
                print("|Info|: ", size_info)
                print("|mu|: ", size_mu)
                break
            if self.cur_cost > prev_cost:
                print("--------------------------------")
                print(f"|  Cost not reduced from {prev_cost} to {self.cur_cost}  |")
                print(f"|  After {n_iters} iterations!  |")
                print("--------------------------------")
                break

            # L, D, _ = scipy.linalg.ldl(new_information)
            self.information = new_information.copy()
            self.mean += delta_mu

            for x_k in self.factored_states:
                x_k.update_factor(total_mean=self.mean.copy(), total_information=self.information.copy())

            n_iters += 1
            prev_cost = self.cur_cost

            if self.debug:
                print( "########################################")
                print("Iteration: ", n_iters)
                print( "----------------------------------------")
                print(f"Cost: ", self.cur_cost)
                # print(f"mu_{n_iters}: \n", self.mean.T)
                # print(f"Info_{n_iters}: \n", self.information)
                print( "########################################\n")

    
    def get_estimate_list(self):
        est_list = []
        for x_k in factored_state_list:
            x_k:FactoredState
            mean = x_k.get_mean()
            covar = x_k.get_covariance()
            stamp = x_k.stamp
            state_k = VectorState(value=mean, stamp=stamp)
            est_k = StateWithCovariance(state=state_k, covariance=covar)
            est_list.append(est_k)
        
        return est_list

if __name__=="__main__":
    NOISE_ON = True
    LINEAR = False
    GH_DEG = 3
    SIM_TIME = 0.5

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
    # sigma_acc_continuous = 3
    dt = 1 / imu_freq
    R_k = np.array([0.1])
    if LINEAR:
        laser_range = LaserRangeFinder(R_d=R_k)
    else:
        laser_range = NonLinearLaserRangeFinder(R_d=R_k, height=2, distance=7)

    x0_val = [5, 0]
    Simulation = Simulator(t_end=SIM_TIME, freq=imu_freq, x0=x0_val)
    # Set Forcing Function
    # Forcing function f(t) = A sin(wt)
    f = lambda t: 1 * np.sin(2*np.pi*t)
    Simulation.set_forcing_function(f)
    # Generating ground truth
    true_pos, true_vel, true_acc = Simulation.generate_ground_truth()

    _,_,_ = Simulation.generate_measurements(sigma_acc=sigma_acc_continuous, pos_freq=laser_range_freq, acc_freq=imu_freq, meas_model=laser_range)
    ##############################

    # %%
    ######### GVI SETUP ############
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
        np.random.seed(1)
        x0_state = x0_state.plus(nav.randvec(P0))

    x0 = StateWithCovariance(state=VectorState(value=x0_state.value, stamp=gt_data[0].stamp), covariance=P0)
    # gt_data = gt_data[0:2]
    # input_data = input_data[0:2]
    # meas_data = meas_data[0:1]
    # gt_data = gt_data[0:11]
    # input_data = input_data[0:11]
    # meas_data = meas_data[0:2]
    factored_state_list = []
    factored_stamp_list = []
    #############################

    # Define prior
    proj_0 = np.zeros((state_dof, len(input_data)*state_dof))
    proj_0[:, :state_dof] = np.eye(state_dof)
    prior_factor = PriorFactor(mean=x0_state.value, covariance=x0.covariance, proj_matrix=proj_0, stamp=x0_state.stamp, gh_degree=GH_DEG)
    prior_factor.link_prior(x0=x0)
    factored_state_list.append(prior_factor)
    factored_stamp_list.append(x0_state.stamp)

    # Define process prior
    proj_proc_empty = np.zeros((2*state_dof, len(input_data)*state_dof))
    proj_idx = 0
    proj_proc = np.eye(len(input_data)*state_dof)
    x_k_1 = x0_state.copy()
    P_k_1 = P0
    for i in range(len(input_data)-1):
        u_k_1:Input = input_data[i]
        u_k:Input = input_data[i+1]
        dt_u = u_k.stamp - u_k_1.stamp
        x_k = proc_model.evaluate(x_k_1, u=u_k_1, dt=dt_u)
        A_k = proc_model.jacobian(x_k_1, u=u_k_1, dt=dt_u)
        Q_k = proc_model.covariance(x=x_k_1, u=u_k_1, dt=dt_u)
        P_k = A_k @ P_k_1 @ A_k.T + Q_k
        p_mean = np.vstack((x_k_1.value.reshape((-1,1)), x_k.value.reshape((-1,1))))
        Q_k = np.zeros((state_dof, state_dof))
        P_covar = np.block([[P_k_1, Q_k],[Q_k, P_k]])
        proj_k = proj_proc_empty.copy()
        proj_k[:, proj_idx:proj_idx+(2*state_dof)] = np.eye(2*state_dof)
        p_fac_k = ProcessFactor(mean=p_mean, covariance=P_covar, proj_matrix=proj_k, stamp=u_k.stamp, gh_degree=GH_DEG)
        p_fac_k.link_dependent_state(process_model=proc_model, u_k_1 = u_k_1)
        factored_state_list.append(p_fac_k)
        factored_stamp_list.append(u_k.stamp)
        proj_idx += state_dof
        x_k_1 = x_k.copy()
        P_k_1 = P_k.copy()

    # Define measurement factors
    proj_meas_empty = np.zeros((state_dof, len(input_data)*state_dof))
    proj_idx = 0
    for i in range(len(meas_data)):
        meas:Measurement = meas_data[i]
        # TODO: Fix assumption that measurement is synchronized with inputs
        if meas.stamp<=input_data[-1].stamp:
            x_k_idx = factored_stamp_list.index(meas.stamp)
            x_k:FactoredState = factored_state_list[x_k_idx]
            proj_k = proj_meas_empty.copy()
            proj_k[:, proj_idx:proj_idx+state_dof] = np.eye(state_dof)
            meas_factor = MeasurementFactor(mean=x_k.get_mean(), covariance=x_k.get_covariance(), stamp=meas.stamp, proj_matrix=proj_k, gh_degree=GH_DEG)
            meas_factor.link_measurement(meas=meas)
            factored_state_list.append(meas_factor)

    
    # %%
    ######### GVI INIT ##################
    gvi = GVI(factored_states=factored_state_list, total_dim=state_dof*len(gt_data), debug=True)
    # %%
    ######### RUN GVI  ##################
    gvi.solve()
    # %%
    ######### PLOT GVI  ##################
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
    plt.show()

    fig, ax = plt.subplots(2, 1)
    fig.tight_layout()
    ax[0].plot(results_gvi.stamp, results_gvi.value[:, 0], label="ESGVI")
    ax[0].plot(results_gvi.stamp, results_gvi.value_true[:, 0], label="Ground truth")
    ax[0].set_xlabel("t (s)")
    ax[0].set_ylabel("x (m)")
    ax[1].plot(results_gvi.stamp, results_gvi.value[:, 1], label="ESGVI")
    ax[1].plot(results_gvi.stamp, results_gvi.value_true[:, 1], label="Ground truth")
    ax[1].set_xlabel("t (s)")
    ax[1].set_ylabel("v (m/s)")
    ax[0].legend()

    ax[1].legend()
    plt.show()
    # %%
    # Batch Comparison
    estimator = nav.BatchEstimator(solver_type="GN", max_iters=20, step_tol=None, gradient_tol=1e-7, ftol=1e-8, verbose=True)
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
    # ax[0].fill_between(results_gvi.stamp, results_gvi.three_sigma[:, 0], -results_gvi.three_sigma[:, 0], alpha=0.1, color='orange')
    ax[1].set_title("Velocity Error")
    ax[1].plot(results_gvi.stamp, results_gvi.error[:, 1], label='ESGVI', linestyle='--')
    # ax[1].fill_between(results_gvi.stamp, results_gvi.three_sigma[:, 1], -results_gvi.three_sigma[:, 1], alpha=0.1, color='orange')
    ax[1].set_xlabel("Time (s)")
    ax[1].set_ylabel(r'$\dot{r}$ (m/s)')
    ax[0].legend(loc='upper right')
    ax[1].legend()
    plt.tight_layout()
    plt.show()
    fig.savefig('/home/astirl/Documents/courses/assignments/mech_642/final_report/ieeeconf/figs/three_sigma_dual.pdf')

    fig, ax = plt.subplots(2, 1)
    fig.tight_layout()
    ax[0].plot(results_map.stamp, results_map.value[:, 0], label="MAP")
    ax[0].plot(results_map.stamp, results_map.value_true[:, 0], label="Ground truth")
    ax[0].set_xlabel("t (s)")
    ax[0].set_ylabel("x (m)")
    ax[1].plot(results_map.stamp, results_map.value[:, 1], label="MAP")
    ax[1].plot(results_map.stamp, results_map.value_true[:, 1], label="Ground truth")
    ax[1].set_xlabel("t (s)")
    ax[1].set_ylabel("v (m/s)")
    ax[0].legend()
    ax[1].legend()
    plt.show()

    # %%
    
    

# %%

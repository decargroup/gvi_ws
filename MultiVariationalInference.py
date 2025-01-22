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

class FactoredState:
    def __init__(self, state_covar = None, info = None, proj_matrix=None, stamp = None, gh_degree = 3):
        self.state_covar:StateWithCovariance = state_covar
        self.information:np.ndarray = info
        self.sqrt_cov = np.linalg.cholesky(self.state_covar.covariance)
        # Internal class variables
        self._gh_degree = gh_degree
        self._dof = self.state_covar.state.dof

        # Phi functions
        self.can_eval_init = False
        self.can_eval_input = False
        self.can_eval_meas = False
        self.type = "FactoredState: "
        self.projection_matrix:np.ndarray = proj_matrix
        self.stamp = stamp
        
        self._generate_init_sigma_pts(gh_degree=gh_degree)
        self.generate_new_sigma_pts()
        return
    
    def set_meas_phi(self, y:Measurement):
        self.can_eval_meas = True
        self.type += "meas"
        #TODO: Adjust for other covariances
        self._R = y.model.covariance(None).reshape((-1,1))
        self._R_inv = np.linalg.inv(self._R)
        self._meas_model = y.model
        self._y_value = y.value.reshape((-1,1))
        return
    
    def set_input_phi(self, u:Input, process_model:ProcessModel, prev_state:'FactoredState'):
        self.can_eval_input = True
        self.projection_matrix = np.eye(2*self._dof)
        self.type += "input"
        self._u = u.copy()
        self._u.value = u.value.copy()
        self._process_model = process_model
        self._prev_state = prev_state.state_covar.state.copy()
        self._prev_state_stamp = prev_state.stamp
        self._prev_covar = prev_state.state_covar.covariance
        self._prev_info = scipy.linalg.inv(self._prev_covar)
        self._dt = self.stamp - prev_state.stamp
        self._Q = process_model.covariance(x=prev_state.state_covar.state, u=u, dt=self._dt).copy()
        self._Q_inv = scipy.linalg.inv(self._Q)
        self.generate_prior_sigma_pts()

        return
    
    def set_init_phi(self, x0:StateWithCovariance):
        self.can_eval_init = True
        self.type += "init"
        self._x0_check_val = x0.state.value.reshape((-1,1))
        self._P0_check = x0.covariance.copy()
        self._P0_check_inv = np.linalg.inv(x0.covariance)
        # self._P0_check_inv = (np.linalg.inv(x0.covariance))
        
    def update_factored_state(self, total_mean:np.ndarray, information:np.ndarray):
        P = force_PSD(scipy.linalg.inv(information))
        if self.can_eval_input:
            self.state_covar.state.value = (self.projection_matrix @ total_mean)[self._dof:].ravel()
            self.information = (self.projection_matrix @ information @ self.projection_matrix.T)[self._dof:, self._dof:]
            self.state_covar.covariance = (self.projection_matrix @ P @ self.projection_matrix.T)[self._dof:, self._dof:]
        else:
            self.state_covar.state.value = (self.projection_matrix @ total_mean).ravel()
            self.information = (self.projection_matrix @ information @ self.projection_matrix.T)
            self.state_covar.covariance = self.projection_matrix @ P @ self.projection_matrix.T
                
        self.generate_new_sigma_pts()
        return
    
    def update_prev_state(self, prev_state:'FactoredState'):
        # self._prev_state_stamp = prev_state.stamp
        self._prev_state.value = prev_state.state_covar.state.value.ravel()
        self._prev_covar = prev_state.state_covar.covariance
        self._prev_info = prev_state.information
        self.generate_prior_sigma_pts()
        return
    
    def _generate_init_sigma_pts(self, gh_degree:int):
        self._unit_sigma_pts, self._weights = gh_cubature(order_p=gh_degree, state_dof=self._dof)

        # TODO: Fix the prior sigma points state_dof
        self._unit_prior_sigma_pts, self._prior_weights = gh_cubature(order_p=gh_degree, state_dof=2*self._dof)
        return
    
    def generate_new_sigma_pts(self):
        self._sqrt_cov = np.linalg.cholesky(self.state_covar.covariance)
        self._sigma_pts = [self.state_covar.state.value.reshape((-1,1)) + self._sqrt_cov @ sp_i.reshape((-1,1)) 
                for sp_i in self._unit_sigma_pts]
        return
    
    def generate_prior_sigma_pts(self):
        P_k = self.state_covar.covariance
        new_cov = block_diag(self._prev_covar, P_k)
        # TODO: Change size of new_mean, second entrance
        new_mean = np.vstack((self._prev_state.value.reshape((-1,1)), self.state_covar.state.value.reshape((-1,1))))
        sqrt_cov = np.linalg.cholesky(new_cov)
        
        self._prior_sigma_pts = [new_mean + sqrt_cov @ sp_i.reshape((-1,1)) 
                for sp_i in self._unit_prior_sigma_pts]
        
        # new_cov = self._prev_covar
        # new_mean = self._prev_state.value.reshape((-1,1))
        # sqrt_cov = np.linalg.cholesky(new_cov)
        # self._prior_sigma_pts = [new_mean + sqrt_cov @ sp_i.reshape((-1,1)) 
        #         for sp_i in self._unit_sigma_pts]
        return
    
    def phi_dx(self):
        if self.can_eval_input:
            info = block_diag(self._prev_info, self.information)
            return info @ self._expect_mu_phi()
        else:
            return self.information @ self._expect_mu_phi()
    
    def phi_dx_dx(self):
        if self.can_eval_input:
            info = block_diag(self._prev_info, self.information)
            a = info @ self._expect_mu_mu_phi() @ info
            b = info * self._expect_phi()
            return a - b
        else:
            a = self.information @ self._expect_mu_mu_phi() @ self.information
            b = self.information * self._expect_phi()
            return a - b
    
    def phi_dinfo(self):
        if self.can_eval_input:
            info = block_diag(self._prev_info, self.information)
            covar = block_diag(self._prev_covar, self.state_covar.covariance)
            a = -0.5 * self._expect_mu_mu_phi()
            b = 0.5 * covar * self._expect_phi()
            c = 0.5 * covar
            return a + b + c
        else:
            a = -0.5 * self._expect_mu_mu_phi()
            b = 0.5 * self.state_covar.covariance * self._expect_phi()
            c = 0.5 * self.state_covar.covariance
            return a + b + c

    def _expect_mu_mu_phi(self):
        expect = np.zeros_like(self.information)
        mu = self.state_covar.state.value.reshape((-1,1))
        for i, w in enumerate(self._weights):
            expect += w * (self._sigma_pts[i] - mu) @ (self._sigma_pts[i] - mu).T  * self._eval_phi(self._sigma_pts[i])
        if self.can_eval_input:
            z_block = np.zeros((self._dof, self._dof))
            expect = np.block([[z_block, z_block],[z_block, expect.copy()]])
            mu_p = np.vstack((self._prev_state.value.reshape((-1,1)), self.state_covar.state.value.reshape((-1,1))))
            for i, w in enumerate(self._prior_weights):
                phi = self._eval_input_phi(self._prior_sigma_pts[i])
                expect += w * (self._prior_sigma_pts[i] - mu_p) @ (self._prior_sigma_pts[i]- mu_p).T * phi 
        return expect
    
    def _expect_mu_phi(self):
        mu = self.state_covar.state.value.reshape((-1,1))
        expect = np.zeros((self._dof, 1))
        # print(self._sigma_pts[0])
        # print(mu)
        # print(self._eval_phi(self._sigma_pts[0]))
        for i, w in enumerate(self._weights):
            phi = self._eval_phi(self._sigma_pts[i])
            expect += w * (self._sigma_pts[i] - mu) * phi
        if self.can_eval_input:
            expect = np.vstack((np.zeros((self._dof, 1)), expect.copy()))
            mu_p = np.vstack((self._prev_state.value.reshape((-1,1)), self.state_covar.state.value.reshape((-1,1))))
            for i, w in enumerate(self._prior_weights):
                phi= self._eval_input_phi(self._prior_sigma_pts[i])
                expect += w * (self._prior_sigma_pts[i] - mu_p) * phi 
        return expect

    def _expect_phi(self):
        # Scalar
        expect = 0
        for i, w in enumerate(self._weights):
            expect += w * self._eval_phi(self._sigma_pts[i]) 
        
        if self.can_eval_input:
            
            for i, w in enumerate(self._prior_weights):
                phi = self._eval_input_phi(self._prior_sigma_pts[i])
                expect += w * phi 
        
        return expect

    def _eval_phi(self, x:np.ndarray):
        state_x = VectorState(value=x, stamp=self.stamp)
        phi = np.zeros((1,1))
        if self.can_eval_init:
            phi += 0.5 * (self._x0_check_val - state_x.value.reshape((-1,1))).T @ self._P0_check_inv @ (self._x0_check_val - state_x.value.reshape((-1,1)))
            # phi += 0.5 * np.log((2*np.pi)**self._dof * np.linalg.det(self._P0_check))
            

        if self.can_eval_meas:
            phi += 0.5 * (self._y_value - self._meas_model.evaluate(state_x).reshape((-1,1))).T @ self._R_inv @ (self._y_value - self._meas_model.evaluate(state_x).reshape((-1,1))) 
            # phi += 0.5 * np.log((2*np.pi) * np.linalg.det(self._R))

        # if self.can_eval_input:
            
        #     # phi += 0.5 * (self._process_model.evaluate(self._prev_state, self._u, self._dt).value.reshape((-1,1))- x.reshape((-1,1))).T @ self._Q_inv @ (self._process_model.evaluate(self._prev_state, self._u, self._dt).value.reshape((-1,1))- x.reshape((-1,1))) 
        #     phi += 0.5 * (x.reshape((-1,1)) - self._process_model.evaluate(self._prev_state, self._u, self._dt).value.reshape((-1,1))).T @ self._Q_inv @ (x.reshape((-1,1)) - self._process_model.evaluate(self._prev_state, self._u, self._dt).value.reshape((-1,1))) 
        #     # phi += 0.5 * np.log((2*np.pi)**self._dof * np.linalg.det(self._Q))

        # print("phi(x): ", phi)
        return phi

    def _eval_input_phi(self, sp:np.ndarray):
        sp_x_k_1 = sp.reshape((-1,1))[0:self._dof]
        sp_x_k = sp.reshape((-1,1))[self._dof:]
        x_k_1_state = VectorState(value=sp_x_k_1, stamp=self.stamp - self._dt)
        proc_model_val = self._process_model.evaluate(x_k_1_state, self._u, dt=self._dt).value.reshape((-1,1))
        phi = 0.5 * (sp_x_k - proc_model_val).T @ self._Q_inv @ (sp_x_k - proc_model_val) 
        return phi
    
    def __str__(self):
        return self.type
    


class GVI:
    def __init__(self, meas_data:List[Measurement], input_data:List[Input], x0:StateWithCovariance, process_model:ProcessModel, gh_degree:int = 3, debug = False):
        self._gh_degree = gh_degree
        self.debug = debug
        # Sort Data
        self.input_data = input_data
        self.meas_data = meas_data
        self.input_data.sort(key=lambda x: x.stamp)
        self.meas_data.sort(key=lambda x: x.stamp)

        # We want to generate state estimates at
        # each input and measurement timestamp
        self._input_stamps = [round(u.stamp, 4) for u in input_data]
        self._meas_stamps = [round(meas.stamp, 4) for meas in meas_data]
        self._meas_stamps = [m for m in self._meas_stamps if m <= self._input_stamps[-1]]
        self._x0_stamp = [x0.stamp]
        stamps = self._input_stamps + self._meas_stamps + self._x0_stamp

        # Get unique stamps
        self.stamps = list(np.unique(np.array(stamps)))
        # Variables for sizing
        self.num_states = len(self.stamps)
        self.state_dim = x0.state.dof
        self.projection_empty = np.zeros((self.state_dim, self.state_dim*self.num_states))

        # Get inital GVI Estimate
        # Dictionary {t_k: FactoredState}
        self.gvi_est = {}
        self.mean = np.zeros((self.state_dim*self.num_states,1))
        self.information = np.eye(self.num_states*self.state_dim, self.num_states*self.state_dim)
        self.initialise_estimate(x0, input_data, process_model)
        # Create Phi(x)
        self.set_phi_factors(x0, meas_data, input_data, process_model)


        return
    
    def iterate(self):
        n_iters = 0
        prev_cost = np.inf
        if self.debug:
            print("Starting Conditions: ")
            print("mu_{0}: \n", self.mean.T)
            print("Info_{0}: \n", self.information)
        while(True):
            # Update info
            phi_dx = np.zeros_like(self.mean)
            phi = np.zeros((1,1))
            new_information = np.zeros_like(self.information)
            functional_dinfo = np.zeros_like(self.information)
            for t_k in self.stamps:
                factored_state_k:FactoredState = self.gvi_est[t_k]
                # factored_state_k.generate_new_sigma_pts()
                proj_k = factored_state_k.projection_matrix
                phi_dx += proj_k.T @ factored_state_k.phi_dx()
                new_information += proj_k.T @ force_PSD(factored_state_k.phi_dx_dx()) @ proj_k
                # new_information += proj_k.T @ factored_state_k.phi_dx_dx() @ proj_k
                functional_dinfo += proj_k.T @ factored_state_k.phi_dinfo() @ proj_k
                phi += factored_state_k._expect_phi()
            
            self.cur_cost = phi + 0.5 * np.log(scipy.linalg.det(self.information))
            new_information = force_PSD(new_information)
            # new_information = new_information
            # delta_mu = scipy.linalg.solve(new_information, -phi_dx)            
            delta_mu = np.linalg.solve(new_information, -phi_dx)
            delta_info = -2 * self.information @ functional_dinfo @ self.information
            # Calculate breaking condition
            
            size_mu = np.abs(np.linalg.norm(delta_mu))
            # size_info = np.linalg.norm(new_information - self.information)
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
            for t_k in self.stamps:
                factored_state_k:FactoredState = self.gvi_est[t_k]
                factored_state_k.update_factored_state(total_mean=self.mean, information=self.information)
                
                if factored_state_k.can_eval_input:
                    prev_stamp = factored_state_k._prev_state_stamp
                    factored_state_k.update_prev_state(self.gvi_est[prev_stamp])

                # update connecting factored (input) states of current state
                self.gvi_est[t_k] = factored_state_k
            
            if self.debug:
                print( "########################################")
                print("Iteration: ", n_iters)
                print( "----------------------------------------")
                print(f"Cost: ", self.cur_cost)
                # print(f"mu_{n_iters}: \n", self.mean.T)
                # print(f"Info_{n_iters}: \n", self.information)
                print( "########################################\n")
            n_iters += 1
            prev_cost = self.cur_cost
       
    
    def initialise_estimate(self, x0:StateWithCovariance, input_data: List[Input], process_model:ProcessModel):
        t_k_1 = x0.stamp
        x_k_1 = x0.state
        P_k_1 = x0.covariance
        S_k_1 = np.linalg.inv(P_k_1)
        proj_k = self.projection_empty.copy()
        proj_count = 0
        proj_k[:,proj_count:proj_count+self.state_dim] = np.eye(self.state_dim)
        self.gvi_est[t_k_1] = FactoredState(state_covar=StateWithCovariance(state=x_k_1, covariance=P_k_1), info=S_k_1, proj_matrix=proj_k, stamp=t_k_1, gh_degree=self._gh_degree)
        self.mean[proj_count:proj_count+self.state_dim] = x_k_1.value.reshape((-1,1))
        self.information[proj_count:proj_count+self.state_dim, proj_count:proj_count+self.state_dim] = S_k_1
        
        for i in range(0, len(self.input_data)-1):
            u_k_1 = self.input_data[i]
            u_k = self.input_data[i+1]
            dt = u_k.stamp - u_k_1.stamp
            x_k = process_model.evaluate(x_k_1, u_k_1, dt)
            x_k.stamp = x_k_1.stamp + dt
            A_k_1 = process_model.jacobian(x_k_1, u_k_1, dt)
            P_k = A_k_1 @ P_k_1 @ A_k_1.T + process_model.covariance(x_k_1, u=u_k_1, dt=dt)
            # P_k = np.eye(self.state_dim)
            S_k = np.linalg.inv(P_k)
            proj_count += self.state_dim
            proj_k = self.projection_empty.copy()
            proj_k[:,proj_count:proj_count+self.state_dim] = np.eye(self.state_dim)
            self.gvi_est[u_k.stamp] = FactoredState(state_covar=StateWithCovariance(state=x_k, covariance=P_k), info=S_k, proj_matrix=proj_k, stamp=u_k.stamp, gh_degree=self._gh_degree)
            self.mean[proj_count:proj_count+self.state_dim] = x_k.value.reshape((-1,1))
            self.information[proj_count:proj_count+self.state_dim, proj_count:proj_count+self.state_dim] = S_k
            # self.information[proj_count-self.state_dim:proj_count, proj_count:proj_count+self.state_dim] = np.eye(self.state_dim)
            # self.information[proj_count:proj_count+self.state_dim,proj_count-self.state_dim:proj_count] = np.eye(self.state_dim)
            x_k_1 = x_k.copy()
        return   
        
    def set_phi_factors(self, x0:StateWithCovariance, meas_data:List[Measurement], input_data:List[Input], process_model:ProcessModel):
        # Initialize with default values 
        # Do Phi x0
        init_stamp = x0.stamp
        factored_state:FactoredState = self.gvi_est[init_stamp]
        factored_state.set_init_phi(x0)
        
        # Do Phi input
        for idx in range(0, len(input_data)-1):
            u_k_1 = input_data[idx]
            prev_stamp = u_k_1.stamp
            cur_stamp = input_data[idx + 1].stamp
            factored_state:FactoredState = self.gvi_est[cur_stamp]
            prev_fac_state:FactoredState = self.gvi_est[prev_stamp]
            factored_state.set_input_phi(u_k_1, process_model, prev_state=prev_fac_state)

        # Do Phi measurement
        for idx in range(0, len(meas_data)):
            meas = meas_data[idx]
            cur_stamp = meas.stamp
            if cur_stamp <= self.stamps[-1]:
                factored_state:FactoredState = self.gvi_est[cur_stamp]
                factored_state.set_meas_phi(meas)

        return
    
    def get_estimate_list(self):
        estimate_list = []
        for t_k in self.stamps:
            factored_state_k:FactoredState = self.gvi_est[t_k]
            state_k = factored_state_k.state_covar.state
            covar_k = factored_state_k.state_covar.covariance
            stamp_k = factored_state_k.stamp
            state_k.stamp = stamp_k
            x_k = StateWithCovariance(state=state_k, covariance=covar_k)
            estimate_list.append(x_k)
        return estimate_list
    
# Globals Declaration
NOISE_ON = True
LINEAR = False
GH_DEG = 3
np.random.seed(1)
### Simulation Setup ###

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
Simulation = Simulator(t_end=0.5, freq=imu_freq, x0=x0_val)
# Set Forcing Function
# Forcing function f(t) = A sin(wt)
f = lambda t: 1 * np.sin(2*np.pi*t)
Simulation.set_forcing_function(f)
# Generating ground truth
true_pos, true_vel, true_acc = Simulation.generate_ground_truth()

_,_,_ = Simulation.generate_measurements(sigma_acc=sigma_acc_continuous, pos_freq=laser_range_freq, acc_freq=imu_freq, meas_model=laser_range)

# %%
## GVI SETUP ##
# Get Navlie formatted data
gt_data, input_data, meas_data = Simulation.get_nav_info()
x0 = VectorState(value=np.array(x0_val), stamp=gt_data[0].stamp)
# sigma_acc_continuous = 100
dt = 1 / imu_freq
Q_d = np.array([[sigma_acc_continuous**2 / dt]])
proc_model = DoubleIntegrator(Q_d)
P0 = np.eye(2) * 1e-5
if NOISE_ON:
    x0 = x0.plus(nav.randvec(P0))
x0 = StateWithCovariance(state=VectorState(value=np.array(x0_val), stamp=gt_data[0].stamp), covariance=P0)
gt_data = gt_data[0:2]
input_data = input_data[0:2]
meas_data = meas_data[0:1]
# gt_data = gt_data[0:11]
# input_data = input_data[0:11]
# meas_data = meas_data[0:2]
# %%
gvi = GVI(meas_data, input_data, x0, process_model=proc_model, debug=True, gh_degree=GH_DEG)
# %%
gvi.iterate()

# %%

np.random.seed(1)
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
fig, ax = nav.plot_error(results_gvi)
ax[0].set_title("Position")
ax[1].set_title("Velocity")
ax[0].set_xlabel("Time (s)")
ax[1].set_xlabel("Time (s)")
# ax[0].set_ylim(-0.01, 0.01)
# ax[1].set_ylim(-0.1, 0.1)
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

fig, ax = nav.plot_error(results_map)
ax[0].set_title("Position")
ax[1].set_title("Velocity")
# ax[0].set_xlabel("Time (s)")
ax[1].set_xlabel("Time (s)")
plt.tight_layout()
plt.show()

fig, ax = plt.subplots(2, 1)
fig.tight_layout()
ax[0].plot(results_map.stamp, results_map.value[:, 0], label="Batch")
ax[0].plot(results_map.stamp, results_map.value_true[:, 0], label="Ground truth")
ax[0].set_xlabel("t (s)")
ax[0].set_ylabel("x (m)")
ax[1].plot(results_map.stamp, results_map.value[:, 1], label="Batch")
ax[1].plot(results_map.stamp, results_map.value_true[:, 1], label="Ground truth")
ax[1].set_xlabel("t (s)")
ax[1].set_ylabel("v (m/s)")
ax[0].legend()
ax[1].legend()
plt.show()
# %%
fig, ax = plt.subplots(2, 1)
fig.tight_layout()
ax[0].plot(results_map.stamp, results_gvi.value[:,0] - results_map.value[:, 0], label="Error")
ax[0].set_xlabel("t (s)")
ax[0].set_ylabel("x (m)")
ax[1].plot(results_map.stamp, results_gvi.value[:,1] - results_map.value[:, 1], label="Error")
ax[1].set_xlabel("t (s)")
ax[1].set_ylabel("v (m/s)")
ax[0].legend()
ax[1].legend()
plt.show()

# %%

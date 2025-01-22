# %%
import numpy as np
import navlie as nav
from scipy import integrate
from scipy import stats
from navlie.types import MeasurementModel, State, Measurement
from navlie.lib.states import VectorState, VectorInput

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
            raise ValueError
        range_measure = np.sqrt(np.square(x.value[0] + self.distance) + np.square(self.height))
        return range_measure
    
    def covariance(self, x: nav.State) -> np.ndarray:
        return self.R

class MassSpringDamperSystem:

    def __init__(self, m, k, c):
        self.m = m
        self.c = c
        self.k = k
        self.f = lambda t: 0
        self.A = np.array([[0, 1],
                           [-self.k/self.m, -self.c/self.m]])
        self.B = np.array([[0],
                           [1/self.m]])
        
    
    def set_force(self, f) -> None:
        self.f = f
    
    def calc_force(self, t):
        return np.array([[self.f(t)]])

    def ode(self, t: float, x:np.ndarray):
        x = x.reshape(2,1)
        x_dot = (self.A @ x) + (self.B @ self.calc_force(t))
        return x_dot.ravel()
    
class Simulator():

    def __init__(self, t_end = 10, freq=100, x0 = [5,0]):
        self.t_start = 0
        self.t_end = t_end
        self.dt = 1 / freq
        self.time = np.arange(self.t_start, t_end, self.dt)
        self.x0 = x0
        self.mass = 1
        self.spring_const = 0.8
        self.damping_const = 0.5
        self.dynamics_model = MassSpringDamperSystem(self.mass, self.spring_const,self.damping_const)

    def set_forcing_function(self, f):
        self.dynamics_model.set_force(f)
        return
    
    def generate_ground_truth(self):
        """
        Get ground truth measurements.

        Returns:
        true_position, true_velocity, true_acceleration
        """
        sol = integrate.solve_ivp(
                        self.dynamics_model.ode,
                        (self.t_start,self.t_end),
                        self.x0,
                        args=(),
                        t_eval=self.time,
                        rtol = 1e-6,
                        atol=1e-6,
                        method='RK45')
        sol_x = sol.y
        self.true_position = np.array(sol_x[0,:])
        self.true_velocity = np.array(sol_x[1,:])
        self.gt_data = []
        for i, pos in enumerate(self.true_position):
            vel = self.true_velocity[i]
            stamp = self.time[i]
            x_k = VectorState(value=np.array([pos, vel]), stamp=stamp)
            self.gt_data.append(x_k)

        acc_list = []
        for i in range(len(self.time)):
            acc = self.dynamics_model.ode(self.time[i], sol_x[:,i])[1]
            acc_list.append(acc)
        self.true_acceleration = np.array(acc_list)

        return self.true_position, self.true_velocity, self.true_acceleration
    
    def generate_linear_measurements(self, sigma_pos, sigma_acc, pos_freq = 100, acc_freq = 100):
        """
        Returns:

        """
        acc_list = []
        pos_list = []
        time_list = []
        sample_every = int((1/self.dt) / pos_freq)
        for i in range(0,len(self.true_position),sample_every):
            noisy_pos = self.true_position[i] + sigma_pos*np.random.randn()
            pos_list.append(noisy_pos)
            time_list.append(self.time[i])

        sample_every = int((1/self.dt) / acc_freq)
        Q_d_sim = sigma_acc**2 / self.dt
        
        for i in range(0, len(self.true_acceleration),sample_every):
            noisy_acc = self.true_acceleration[i] + np.sqrt(Q_d_sim)*np.random.randn()
            acc_list.append(noisy_acc)

        self.measured_position = np.array(pos_list)
        self.measured_acceleration = np.array(acc_list)
        self.measured_time = np.array(time_list)

        return self.measured_position, self.measured_acceleration, self.measured_time
    
    def generate_measurements(self, sigma_acc, 
                              pos_freq, acc_freq, 
                              meas_model:NonLinearLaserRangeFinder):
        """
        Returns:

        """
        self.meas_data = []
        self.input_data = []
        acc_list = []
        pos_list = []
        time_list = []
        sample_every = int((1/self.dt) / pos_freq)
        for i in range(0,len(self.true_position),sample_every):
            meas = meas_model.evaluate(self.gt_data[i])
            pos = meas.ravel()
            noisy_pos = pos + np.sqrt(meas_model.covariance(self.gt_data[i]))*np.random.randn()
            noisy_meas = Measurement(value=np.array([noisy_pos]), stamp=round(self.time[i], ndigits=4), model=meas_model)
            self.meas_data.append(noisy_meas)
            pos_list.append(noisy_pos)
            time_list.append(self.time[i])

        sample_every = int((1/self.dt) / acc_freq)
        Q_d_sim = sigma_acc**2 / self.dt
        
        for i in range(0, len(self.true_acceleration),sample_every):
            noisy_acc = self.true_acceleration[i] + np.sqrt(Q_d_sim)*np.random.randn()
            acc_list.append(noisy_acc)
            u_k = VectorInput(value=np.array([noisy_acc]), stamp=round(self.time[i], ndigits=4), covariance=Q_d_sim)
            self.input_data.append(u_k)

        self.measured_position = np.array(pos_list)
        self.measured_acceleration = np.array(acc_list)
        self.measured_time = np.array(time_list)

        return self.measured_position, self.measured_acceleration, self.measured_time
    
    def get_nav_info(self):
        return self.gt_data, self.input_data, self.meas_data
    
if __name__=="__main__":
    from matplotlib import pyplot as plt
    
    # %%
    sigma_acc_continuous = 0.045
    R_k = 0.01
    laser_range_freq = 10
    imu_freq = 100

    Simulation = Simulator(t_end=20, freq=imu_freq, x0=[5,0])
    # Set Forcing Function
    # Forcing function f(t) = A sin(wt)
    f = lambda t: 1 * np.sin(2*np.pi*t)
    # Change function in mass spring damper
    Simulation.set_forcing_function(f)
    # Generating ground truth
    true_pos, true_vel, true_acc = Simulation.generate_ground_truth()
    t = Simulation.time
    fig, ax = plt.subplots(3,1, sharex=True)
    ax[0].set_ylabel(r'$x(t)$ (m)')
    # ax[0].set_xlabel(r'$t$ [s]')
    ax[0].plot(t,true_pos)

    ax[1].set_ylabel(r'$\dot{x}(t)$ (m/s)')
    # ax[1].set_xlabel(r'$t$ [s]')
    ax[1].plot(t,true_vel)

    ax[2].set_ylabel(r'$\ddot{x}(t)$ (m/s)')
    ax[2].set_xlabel(r'$t$ (s)')
    ax[2].plot(t,true_acc)


    # %%
    meas_model = NonLinearLaserRangeFinder(R_d=R_k, height=2, distance=7)
    measured_pos, measured_acc, measured_time = Simulation.generate_nonlinear_measurements(sigma_acc=sigma_acc_continuous, pos_freq=laser_range_freq, acc_freq=imu_freq, meas_model=meas_model)
    fig, ax = plt.subplots(2,1, sharex=True)
    ax[0].set_ylabel(r'$y(t)$ (m)')
    # ax[0].set_xlabel(r'$t$ (s)')
    ax[0].scatter(measured_time,measured_pos,s=0.05, marker = 'x')

    ax[1].set_ylabel(r'$u^{acc}(t)$ ($\frac{m}{s^2}$)')
    ax[1].set_xlabel(r'$t$ (s)')    
    ax[1].scatter(Simulation.time,measured_acc, s=0.05)
    # %%

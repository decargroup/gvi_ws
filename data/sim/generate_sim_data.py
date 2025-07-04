import numpy as np
import navlie as nav
import matplotlib.pyplot as plt
from typing import List, Tuple
import csv

from gvi_ws.graph.factors import Factor, ProcessFactor, MeasurementFactor, PriorFactor
from gvi_ws.graph.esgvi import ESGVI
from gvi_ws.graph.losses import SkewLaplaceLoss, GaussianLoss, StudentTLoss, CauchyLoss
from gvi_ws.graph.construct_esgvi import generate_esgvi_graph, esgvi_from_map
from gvi_ws.util.map_batch import construct_planar_map
from gvi_ws.util.load_config import load_config

from gvi_ws.util.data_generation import DataGenerator
from navlie.lib.states import SE2State, VectorState, State
from navlie.types import StateWithCovariance, MeasurementModel, ProcessModel, Input
from navlie.lib.models import SingleIntegrator, DoubleIntegrator, RangePointToAnchor
from navlie.lib.states import VectorInput
from navlie.batch.losses import L2Loss, CauchyLoss
CSV_PATH = "./data/sim/"

if __name__ == "__main__":
    # Dataset
    data_config = load_config("config/gen_data.yaml")
    dataset = data_config["dataset"]

    config = load_config(f"config/{dataset}_localization.yaml")
    noise_config = load_config("config/noise_config.yaml")
    
    # Model Params
    proc_model_dict:dict[str, ProcessModel] = {"single_integrator":SingleIntegrator, "double_integrator":DoubleIntegrator}
    meas_model_dict:dict[str, MeasurementModel] = {"range_point": RangePointToAnchor}
    Q0_mult = float(data_config["Q0_mult"])
    P0_mult = float(data_config["P0_mult"])
    R0_mult = float(data_config["R0_mult"])
    x0_pos = data_config["x0_pos"]
    anchor_pos_list = data_config["anchor_pos"]

    dof = 2
    state:State = VectorState
    if dataset == "se2":
        state = SE2State
        dof = 3
    # Noise Config
    Q_d = np.identity(dof) * Q0_mult
    R_d = np.identity(1) * 1e-3
    # Init Prior
    x0 = state(value=np.array([1, 0]), stamp=0.0, state_id="x0")
    P0 = np.identity(dof) * P0_mult

    meas_model = meas_model_dict[data_config["meas_model"]]
    meas_model = [meas_model(anchor_position=anchor_pos, R=R_d) for anchor_pos in anchor_pos_list]
    meas_freq = 10
    proc_model = proc_model_dict[data_config["proc_model"]]
    proc_model = proc_model(Q=Q_d)
    proc_freq = 100

    # Data Generation
    input_profile = lambda t, x: np.array([np.sin(t), np.cos(t)])

    # Data Params
    np.random.seed(config["SEED"])
    T_END = config["T_END"]
    NOISE = config["NOISE"]

    # Noise Params
    PROC_NOISE = noise_config["PROC_NOISE"]
    MEAS_NOISE = noise_config["MEAS_NOISE"]

    # Script Params
    SAVE_FIGS = data_config["save_figs"]
    SHOW_FIGS = data_config["show_figs"]

    # Gaussian Data Generation
    dg_gaussian = DataGenerator(
        process_model=proc_model,
        input_func=input_profile,
        input_covariance=Q_d,
        input_freq=proc_freq,
        meas_model_list=meas_model,
        meas_freq_list=meas_freq,
        process_noise_type="gaussian",
        measurement_noise_type="gaussian",
    )
    # Other Heavy-Tailed Noise Generation
    dg_heavy = DataGenerator(
        process_model=proc_model,
        input_func=input_profile,
        input_covariance=Q_d,
        input_freq=proc_freq,
        meas_model_list=meas_model,
        meas_freq_list=meas_freq,
        process_noise_type=PROC_NOISE,
        measurement_noise_type=MEAS_NOISE,
    )

    gt_data, input_data_gauss, meas_data_gauss = dg_gaussian.generate(
        x0.copy(), 0, T_END, noise=NOISE
    )
    _, input_data_heavy, meas_data_heavy = dg_heavy.generate(
        x0.copy(), 0, T_END, noise=NOISE
    )
    fig, axs = plt.subplots(1, 2, sharey=True)
    fig_gauss, ax_gauss = nav.plot_meas(meas_data_gauss, state_list=gt_data, axs=axs[0])
    ax_gauss[0].set_title(f"Gaussian Range Measurements")
    ax_gauss[0].set_xlabel(f"Time (s)")
    ax_gauss[0].set_ylabel(f"Range (m)")
    fig_dual, ax_heavy = nav.plot_meas(meas_data_heavy, state_list=gt_data, axs=axs[1])
    ax_heavy[0].set_title(f"{MEAS_NOISE.capitalize()} Range Measurements")
    ax_heavy[0].set_xlabel(f"Time (s)")
    low, up = ax_heavy[0].get_ylim()
    ax_gauss[0].set_ybound(low, up)
    fig.tight_layout()
    if SAVE_FIGS:
        plt.savefig(
            f"/home/astirl/Documents/courses/assignments/mech_642/gvi_ws/figs/{dataset}_noise_comp.pdf"
        )
    if SHOW_FIGS:
        plt.show()
    # Save inputs
    with open(CSV_PATH + f"{dataset}_input_data.csv", "w", newline="") as f:
        writer = csv.writer(f)
        if dof == 2:
            writer.writerow(["stamp", "state_id", "v_x", "v_y"])  # Extend if >2D
        else:
            writer.writerow(["stamp", "state_id", "w", "v_x", "v_y"])  # Extend if >2D
        
        for inp in input_data_heavy:
            writer.writerow([
                inp.stamp,
                inp.state_id if inp.state_id is not None else "",
                *inp.value # assuming vec is a 1D array-like
            ])

    # Save measurements
    with open(CSV_PATH + f"{dataset}_meas_data.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["stamp", "state_id", "measurement_type", "value"])
        for meas in meas_data_heavy:
            writer.writerow([
                meas.stamp,
                meas.state_id if meas.state_id is not None else "",
                type(meas.model).__name__,
                meas.value if hasattr(meas, 'value') else meas.value])
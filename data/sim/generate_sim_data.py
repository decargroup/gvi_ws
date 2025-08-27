import numpy as np
import navlie as nav
import matplotlib.pyplot as plt
from typing import List, Tuple
from scipy.interpolate import interp1d
import csv
import os
import pickle
import seaborn as sns

from gvi_ws.graph.factors import Factor, ProcessFactor, MeasurementFactor, PriorFactor
from gvi_ws.graph.esgvi import ESGVI
from gvi_ws.graph.losses import SkewLaplaceLoss, GaussianLoss, StudentTLoss, CauchyLoss
from gvi_ws.graph.construct_esgvi import generate_esgvi_graph, esgvi_from_map
from gvi_ws.util.map_batch import construct_planar_map
from gvi_ws.util.load_config import load_config

from gvi_ws.util.data_generation import DataGenerator
from navlie.lib.states import SE2State, VectorState, State
from navlie.types import (
    StateWithCovariance,
    MeasurementModel,
    ProcessModel,
    Input,
    Measurement,
)
from navlie.lib.models import (
    SingleIntegrator,
    DoubleIntegrator,
    BodyFrameVelocity,
    RangePointToAnchor,
    RangePoseToAnchor,
)
from navlie.lib.states import VectorInput
from navlie.batch.losses import L2Loss, CauchyLoss
from gvi_ws.util.fit_skew_laplace import fit_skew_laplace, skew_laplace_pdf
from scipy.stats import norm, cauchy
from sklearn.mixture import GaussianMixture


def compute_meas_error(
    meas_list: List[Measurement], state_list: List[State]
) -> Tuple[plt.Figure, plt.Axes, np.ndarray]:
    """
    Given measurement data, make histogram plots of the measurement errors
    using the ground-truth model-based values.

    Parameters
    ----------
    meas_list : List[Measurement]
        Measurement data to be plotted.
    state_list : List[State]
        A list of true State objects with similar timestamp domain. Will be
        interpolated if timestamps do not line up perfectly.
    """
    # Convert everything to numpy arrays for plotting, and compute the
    # ground-truth model-based measurement value.

    meas_list.sort(key=lambda y: y.stamp)

    # Find the state of the nearest timestamp to the measurement
    y_stamps = np.array([y.stamp for y in meas_list])
    x_stamps = np.array([x.stamp for x in state_list])
    indexes = np.array(range(len(state_list)))
    nearest_state = interp1d(
        x_stamps,
        indexes,
        "nearest",
        bounds_error=False,
        fill_value="extrapolate",
    )
    state_idx = nearest_state(y_stamps)
    y_meas = []
    y_true = []
    for i in range(len(meas_list)):
        data = np.ravel(meas_list[i].value)
        y_meas.append(data)
        x = state_list[int(state_idx[i])]
        y = meas_list[i].model.evaluate(x)
        if y is None:
            y_true.append(np.zeros_like(data) * np.nan)
        else:
            y_true.append(np.ravel(y))
            R = np.atleast_2d(meas_list[i].model.covariance(x))

    y_meas = np.atleast_2d(np.array(y_meas))
    y_true = np.atleast_2d(np.array(y_true))
    error_meas = y_meas - y_true
    # Plot histogram
    fig, ax = plt.subplots()
    ax.hist(
        error_meas,
        bins="fd",
        alpha=0.6,
        color="grey",
        edgecolor="black",
        density=True,
    )
    # ax.set_title("Histogram of Measurement Errors")
    ax.set_xlabel("Error (m)")
    ax.set_ylabel("Probability Density")
    fig.tight_layout()

    return fig, ax, error_meas


EXP_PATH = "./data/sim/"

if __name__ == "__main__":
    # Plotting parameters
    plt.rc("text", usetex=True)
    plt.rc("font", family="serif", size=14)
    plt.rc("lines", linewidth=2)
    plt.rc("axes", grid=True)
    plt.rc("grid", linestyle="--")
    sns.set_theme(style="whitegrid")
    # Dataset
    gen_data_config = load_config("config/gen_data.yaml")
    dataset = gen_data_config["dataset"]

    config = load_config(f"config/{dataset}_localization.yaml")
    noise_config = load_config("config/noise_config.yaml")

    # Model Params
    proc_model_dict: dict[str, ProcessModel] = {
        "single_integrator": SingleIntegrator,
        "double_integrator": DoubleIntegrator,
        "body_frame_velocity": BodyFrameVelocity,
    }
    meas_model_dict: dict[str, MeasurementModel] = {
        "range_point": RangePointToAnchor,
        "range_pose": RangePoseToAnchor,
    }
    Q0_mult = gen_data_config["Q0_mult"]
    P0_mult = float(gen_data_config["P0_mult"])
    R0_mult = float(gen_data_config["R0_mult"])
    x0_pos = gen_data_config["x0_pos"]
    anchor_pos_list = gen_data_config["anchor_pos"]

    dof = 2
    state: State = VectorState
    if dataset == "se2":
        state = SE2State
        dof = 3

    # Noise Config
    Q_d = np.diag(np.array(Q0_mult))
    R_d = np.identity(1) * R0_mult
    sigma_true = np.sqrt(R_d)
    # Init Prior
    x0 = state(value=np.array(x0_pos), stamp=0.0, state_id="x0")
    P0 = np.identity(dof) * P0_mult

    meas_model_str = gen_data_config["meas_model"]
    meas_model = meas_model_dict[meas_model_str]
    if meas_model_str == "range_point":
        meas_model = [
            meas_model(anchor_position=anchor_pos, R=R_d)
            for anchor_pos in anchor_pos_list
        ]
    elif meas_model_str == "range_pose":
        tag_pos = np.array(gen_data_config["tag_pos"])
        meas_model = [
            meas_model(anchor_position=anchor_pos, tag_body_position=tag_pos, R=R_d)
            for anchor_pos in anchor_pos_list
        ]
    meas_freq = 10
    proc_model = proc_model_dict[gen_data_config["proc_model"]]
    proc_model = proc_model(Q=Q_d)
    proc_freq = 100

    # Data Generation
    input_profile = lambda t, x: np.array([np.sin(t), np.cos(t)])
    if dataset == "se2":
        # Input Profile
        input_profile = lambda t, x: np.array([np.cos(0.1 * t), 1.0, 0])

    # Data Params
    np.random.seed(config["SEED"])
    MAX_TIME = gen_data_config["max_time"]
    NOISE = config["NOISE"]

    # Noise Params
    PROC_NOISE = noise_config["PROC_NOISE"]
    MEAS_NOISE = noise_config["MEAS_NOISE"]
    skew_lambda_gt = float(noise_config["GVI_SKEW_LAMBDA"])

    # Script Params
    SAVE_FIGS = gen_data_config["save_figs"]
    SHOW_FIGS = gen_data_config["show_figs"]
    EXPORT = gen_data_config["export"]

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
        x0.copy(), 0, MAX_TIME, noise=NOISE
    )
    _, input_data_heavy, meas_data_heavy = dg_heavy.generate(
        x0.copy(), 0, MAX_TIME, noise=NOISE
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

    fig_error, axs_error, meas_errors = compute_meas_error(
        meas_list=meas_data_heavy, state_list=gt_data
    )
    # Fit distributions
    mu, std = norm.fit(meas_errors)
    mu_c, std_c = cauchy.fit(meas_errors)
    mu_sl, std_sl, lambda_sl = fit_skew_laplace(meas_errors)
    # Mixture Model
    gmm = GaussianMixture(n_components=3, covariance_type="diag", random_state=0)
    gmm.fit(np.array(meas_errors).reshape((-1, 1)))

    # X-range for PDFs
    x = np.linspace(-1, 2, 500)
    pdf_gauss = norm.pdf(x, mu, std)
    pdf_cauchy = cauchy.pdf(x, mu_c, std_c)
    # Compute skew laplace pdf
    pdf_sl = skew_laplace_pdf(x, mu_sl, std_sl, lambda_sl)
    # Compute mixture pdf
    logprob = gmm.score_samples(x.reshape((-1, 1)))
    pdf_gmm = np.exp(logprob)

    # Plot pdfs overtop of histogram
    axs_error.plot(
        x,
        pdf_gauss,
        "--",
        linewidth=2,
        color="tab:blue",
        label=rf"Gaussian Fit\\ $\mu={mu:.2f},\ \sigma={std:.2f}$",
    )
    # axs_error.plot(
    #     x,
    #     pdf_cauchy,
    #     "--",
    #     linewidth=2,
    #     color="tab:orange",
    #     label=f"Cauchy Fit\nμ={mu_c:.2f}, σ={std_c:.2f}",
    # )
    gmm_means = gmm.means_.flatten()
    gmm_stds = np.sqrt(gmm.covariances_.flatten())
    gmm_weights = gmm.weights_
    idx_sort = np.argsort(gmm_weights)[::-1]

    means_str = ", ".join(f"{gmm_means[i]:.2f}" for i in idx_sort)
    stds_str = ", ".join(f"{gmm_stds[i]:.2f}" for i in idx_sort)
    weights_str = ", ".join(f"{gmm_weights[i]:.2f}" for i in idx_sort)

    # Plot GMM PDF
    axs_error.plot(
        x,
        pdf_gmm,
        "-.",
        linewidth=2,
        color="tab:purple",
        label=rf"Gaussian Mixture Fit\\ $\mu=[{means_str}]$\\ $\sigma=[{stds_str}]$\\ $w=[{weights_str}]$",
    )
    # Plot Skew-Laplace PDF
    axs_error.plot(
        x,
        pdf_sl,
        "--",
        linewidth=2,
        color="tab:orange",
        label=rf"Skew-Laplace Fit\\ $\mu={mu_sl:.2f},\ \sigma={std_sl:.2f},\ \lambda={lambda_sl:.3f}$",
    )
    pdf_sl_true = skew_laplace_pdf(x, mu=0, sigma=sigma_true[0, 0], lam=skew_lambda_gt)
    # Plot the true Skew-Laplace PDF
    axs_error.plot(
        x,
        pdf_sl_true,
        "-",
        linewidth=2,
        color="tab:green",
        label=rf"True Noise\\ $\mu=0.00,\ \sigma={sigma_true[0,0]:.2f},\ \lambda={skew_lambda_gt:.3f}$",
    )
    axs_error.legend(fontsize=14, loc="upper right", frameon=True)
    axs_error.set_xlim(left=-0.25, right=1.4)
    # Tick label font size
    axs_error.tick_params(
        axis="both",
        labelsize=10,
    )
    # Axis Labels
    axs_error.set_xlabel(rf"Range Error, $e_r$ (m)", fontsize=14)
    axs_error.set_ylabel("Probability Density", fontsize=13)

    fig_error.tight_layout()

    noise_params = {
        "Gaussian": [mu, std],
        "Cauchy": [mu_c, std_c],
        "GMM": [
            gmm.means_.flatten().tolist(),
            np.sqrt(gmm.covariances_.flatten().tolist()),
            gmm.weights_.tolist(),
        ],
        "Skew Laplace": [mu_sl, std_sl, lambda_sl],
    }
    if SAVE_FIGS:
        fig.savefig(f"figs/{dataset}/se2_noise_comp.pdf")
        fig_error.savefig(f"figs/{dataset}/se2_range_errors.pdf")
    if SHOW_FIGS:
        plt.show()
    if EXPORT:
        # Save inputs
        os.makedirs(EXP_PATH, exist_ok=True)

        save_path = os.path.join(EXP_PATH, f"meas_data_{dataset}.pkl")
        with open(save_path, "wb") as f:
            pickle.dump(
                {
                    "ground_truth": gt_data,
                    "input_data": input_data_heavy,
                    "meas_data_non_gauss": meas_data_heavy,
                    "meas_data_gauss": meas_data_gauss,
                    "process_model": proc_model,
                    "meas_model": meas_model,
                    "x0": x0,
                    "P0": P0,
                    "fitted_noise_params": noise_params,
                    "landmarks": anchor_pos_list,
                },
                f,
            )

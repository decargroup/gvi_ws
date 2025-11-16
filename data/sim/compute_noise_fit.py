import numpy as np
import navlie as nav
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os

from gvi_ws.util.fit_skew_laplace import (
    fit_skew_laplace,
    skew_laplace_pdf,
    fit_skew_laplace_fixed_mu,
    fit_two_piece_cauchy,
    two_piece_cauchy_pdf,
)
from scipy.stats import norm, cauchy, t
from sklearn.mixture import GaussianMixture
from gvi_ws.util.data_generation import randvec
from gvi_ws.util.load_config import load_config

# Plotting parameters
plt.rc("text", usetex=True)
plt.rc("font", family="serif", size=14)
plt.rc("lines", linewidth=2)
plt.rc("axes", grid=True)
plt.rc("grid", linestyle="--")
sns.set_theme(style="whitegrid")

SAVE_FIGS = True
EXPORT = True
# For Monte Carlo Trials, fitting the data with fewer samples messes with the estimator.
# Thi script fits more samples to improve performance

if __name__ == "__main__":
    np.random.seed(1)
    nlos_flag = False
    noise_config = load_config("config/noise_config.yaml")
    noise_gen_method = noise_config["MEAS_NOISE"]
    if noise_gen_method == "nlos":
        noise_gen_method = "gaussian"
        nlos_flag = True
    noise_model_gvi = noise_config["GVI_LOSS_FUN"]
    noise_model_map = noise_config["MAP_LOSS_FUN"]

    gen_data_config = load_config("config/gen_data.yaml")
    covar = np.identity(1) * float(gen_data_config["R0_mult"])
    std_dev = np.sqrt(covar)
    print(std_dev)
    samples = randvec(cov=covar, num_samples=5000, method=noise_gen_method)
    samples = samples.flatten()
    if nlos_flag:
        nlos_percent = 0.25
        nlos_indx = np.random.choice(
            len(samples), int(nlos_percent * len(samples)), replace=False
        )
        for i in nlos_indx:
            nlos_inc = np.random.uniform(1 * std_dev[0, 0], 6 * std_dev[0, 0])
            samples[i] = samples[i] + nlos_inc

    if noise_model_gvi == "skew_laplace":
        mu_sl, std_sl, lambda_sl = fit_skew_laplace_fixed_mu(samples)
    elif noise_model_gvi == "student_t":
        mu_t, std_t, dof_t = t.fit(samples)

    # Fit distributions
    mu, std = norm.fit(samples)

    mu_c, std_c = cauchy.fit(samples)

    mu_c_mod, c_pos, c_neg = fit_two_piece_cauchy(samples)

    # X-range for PDFs
    x = np.linspace(-1, 2.5, 500)
    # Compute skew laplace pdf
    pdf_sl = skew_laplace_pdf(x, mu_sl, std_sl, lambda_sl)
    # Gaussian pdf
    pdf_gauss = norm.pdf(x, mu, std)
    # Cauchy pdf
    pdf_c_mod = two_piece_cauchy_pdf(x, mu_c_mod, c_pos, c_neg)

    # Mixture Model
    gmm = GaussianMixture(n_components=3, covariance_type="diag", random_state=0)
    gmm.fit(np.array(samples).reshape((-1, 1)))
    # Compute mixture pdf
    logprob = gmm.score_samples(x.reshape((-1, 1)))
    pdf_gmm = np.exp(logprob)
    gmm_means = gmm.means_.flatten()
    gmm_stds = np.sqrt(gmm.covariances_.flatten())
    gmm_weights = gmm.weights_
    idx_sort = np.argsort(gmm_weights)[::-1]

    means_str = ", ".join(f"{gmm_means[i]:.2f}" for i in idx_sort)
    stds_str = ", ".join(f"{gmm_stds[i]:.2f}" for i in idx_sort)
    weights_str = ", ".join(f"{gmm_weights[i]:.2f}" for i in idx_sort)

    # Separate altered and unchanged samples
    samples = np.array(samples)
    nlos_mask = np.zeros_like(samples, dtype=bool)
    nlos_mask[nlos_indx] = True

    samples_nlos = samples[nlos_mask]  # altered (red)
    samples_normal = samples[~nlos_mask]  # unchanged (green)

    # --- Plot stacked histogram ---
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.hist(
        [samples_normal, samples_nlos],
        bins="fd",
        stacked=True,
        color=["green", "red"],
        alpha=0.6,
        edgecolor="black",
        label=["LOS", "NLOS/Multipath"],
        density=True,
    )
    ax.set_title("Histogram of Measurement Errors")
    ax.set_xlabel(r"Range Error, $e_r$ (m)")
    ax.set_ylabel("Probability Density")
    # ax.plot(
    #     x,
    #     pdf_gauss,
    #     "-",
    #     linewidth=2,
    #     color="tab:green",
    #     label=rf"Gaussian Fit\\ $\mu={mu:.2f},\ \sigma={std:.2f}$",
    # )
    ax.plot(
        x,
        pdf_c_mod,
        "-.",
        linewidth=2,
        color="tab:blue",
        label=rf"Asymmetric Cauchy Fit\\ $\mu={mu_c:.2f}$,\ $c^-={c_neg:.2f}$,\ $c^+={c_pos:.2f}$",
    )

    # Plot Skew-Laplace PDF
    ax.plot(
        x,
        pdf_sl,
        "-.",
        linewidth=2,
        color="tab:orange",
        label=rf"Skew-Laplace Fit\\ $\mu={mu_sl:.2f},\ \sigma={std_sl:.2f},\ \lambda={lambda_sl:.3f}$",
    )
    # Plot GMM PDF
    ax.plot(
        x,
        pdf_gmm,
        "-.",
        linewidth=2,
        color="tab:purple",
        label=rf"Gaussian Mixture Fit\\ $\mu=[{means_str}]$\\ $\sigma=[{stds_str}]$\\ $w=[{weights_str}]$",
    )
    ax.legend(fontsize=14, loc="upper right", frameon=True)
    ax.set_xlim(left=-1, right=2.5)
    ax.set_ylim(bottom=0, top=1.25)
    if SAVE_FIGS:
        if noise_gen_method == "student_t":
            plt.savefig(f"./figs/monte_carlo/sim_fit_st.pdf")
        elif noise_gen_method == "skew_laplace":
            plt.savefig(f"./figs/monte_carlo/sim_fit_sl.pdf")
        elif nlos_flag:
            plt.savefig(f"./figs/monte_carlo/sim_fit_nlos.pdf")
        else:
            plt.savefig(f"./figs/monte_carlo/sim_fit.pdf")
    plt.show()
    if EXPORT:
        save_path = f"./data/sim/meas_data_se2_sim_{noise_gen_method}.pkl"
        if nlos_flag:
            save_path = f"./data/sim/meas_data_se2_sim_nlos.pkl"
        with open(save_path, "rb") as f:
            data = pickle.load(f)

        print("Original fitted_noise_params:")
        print(data["fitted_noise_params"])

        data["fitted_noise_params"]["GMM"] = [
            gmm.means_.flatten().tolist(),
            np.sqrt(gmm.covariances_.flatten().tolist()),
            gmm.weights_.tolist(),
        ]
        data["fitted_noise_params"]["Skew Laplace"] = [mu_sl, std_sl, lambda_sl]

        data["fitted_noise_params"]["Asymmetric Cauchy"] = [mu_c_mod, c_pos, c_neg]

        print("Updated fitted_noise_params:")
        print(data["fitted_noise_params"])

        # Resave the file with updated fit
        with open(save_path, "wb") as f:
            pickle.dump(data, f)

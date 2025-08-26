# %%
import numpy as np
import scipy.linalg
import navlie as nav
import scipy.stats
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Ellipse
from typing import List, Dict

from gvi_ws.graph.factors import Factor, ProcessFactor, MeasurementFactor, PriorFactor
from gvi_ws.models.models import LaserRangeFinder
from gvi_ws.util.psd import force_sym_PSD
from navlie.types import State, StateWithCovariance, Measurement, Input
from navlie.lib.states import MatrixLieGroupState, SE2State, VectorState
from navlie.filters import generate_sigmapoints
from gvi_ws.util.cubatures import (
    student_t_cubature,
    gh_cubature,
    trans_gh_cubature,
)
from gvi_ws.util.data_generation import randvec
from gvi_ws.util.psd import (
    force_sym_PSD,
    force_sym,
)
from navlie.lib.models import BodyFrameVelocity, LinearMeasurement
from gvi_ws.util.fit_skew_laplace import fit_skew_laplace, skew_laplace_pdf
from scipy.stats import norm, cauchy
from sklearn.mixture import GaussianMixture


def test_student_t_meas_factor(verbose=False, method="gh", order=3):
    key1 = "x0"
    projection = np.identity(2)
    state_list = [VectorState(value=np.ones((2, 1)), stamp=0.0, state_id="x0")]
    var_slices = {"x0": slice(0, state_list[0].dof)}
    R_k = np.identity(2) * 1e-2
    R_k_inv = force_sym(scipy.linalg.inv(R_k))
    meas_model = LinearMeasurement(C=np.identity(2), R=R_k.copy())
    meas_val = meas_model.evaluate(state_list[0])
    meas = Measurement(value=meas_val, stamp=state_list[0].stamp, model=meas_model)
    P0 = np.identity(state_list[0].dof) * 1e-3
    meas_fac = MeasurementFactor(
        keys=key1,
        measurement=meas,
        variable_slices=var_slices,
        projection=projection,
        cubature=method,
        order=order,
    )
    total_covariance = P0.copy()
    total_information = force_sym(scipy.linalg.inv(total_covariance))

    # Test 1: Instance
    assert isinstance(meas_fac, MeasurementFactor)

    # Test 2: Derivative Evaluation
    cost, col, matrix = meas_fac.evaluate_derivatives(
        states=state_list, covar_matrix=total_covariance, info_matrix=total_information
    )

    assert col.shape[0] == total_covariance.shape[0]
    assert matrix.shape == total_covariance.shape
    if verbose:
        print(col)
        print(matrix)
        print(cost)

    delta_mean = scipy.linalg.solve(matrix, -col)
    if verbose:
        print(delta_mean)

    all_close = np.allclose(delta_mean, np.zeros_like(delta_mean))
    print("----------\nTesting mean vector update: ")
    assert (
        all_close
    ), f"Delta_mean is {delta_mean} rather than {np.zeros_like(delta_mean)}."
    print("Passed!")
    print("----------\nTesting information vector update: ")
    assert np.allclose(
        matrix, R_k_inv
    ), f"Info update:\n{matrix} \n Not equal to measurement info:\n {R_k_inv}"
    print("Passed!")


def sample_skew_laplace(mu, sigma, lam, n_samples=10000):
    """
    Samples from the skew-Laplace distribution using the NVMM representation.

    Parameters:
        mu (float): Location parameter
        sigma (float): Scale parameter (> 0)
        lam (float): Skewness parameter
        n_samples (int): Number of samples to draw

    Returns:
        np.ndarray: Array of samples from skew-Laplace distribution
    """
    beta = np.random.gamma(shape=1, scale=2, size=n_samples)  # ~ Gamma(1,2)
    z = np.random.normal(loc=mu + beta * lam, scale=np.sqrt(beta) * sigma)
    return np.atleast_2d(z)


def sample_skew_laplace_mv(mu, cov, lam, num_samples=10000):
    """
    Multivariate skew-Laplace sampling using the NVMM structure with Cholesky-based correlation.

    Parameters:
        mu (np.ndarray): Mean vector (d,)
        cov (np.ndarray): Covariance matrix (d, d)
        lam (np.ndarray): Skewness vector (d,)
        num_samples (int): Number of samples to generate

    Returns:
        np.ndarray: Samples of shape (d, num_samples)
    """
    d = len(mu)
    mu = np.asarray(mu).reshape(-1, 1)
    lam = np.asarray(lam).reshape(-1, 1)

    # Cholesky decomposition of covariance matrix
    L = np.linalg.cholesky(cov)

    # Step 1: Sample beta ~ Gamma(1,2)
    beta = np.random.gamma(shape=1, scale=2, size=(1, num_samples))

    # Step 2: Sample standard normal noise
    eps = np.random.normal(size=(d, num_samples))

    # Step 3: Scale and shift
    z = mu + lam @ beta + L @ (np.sqrt(beta) * eps)

    return z


def plot_cov_ellipse(
    cov, mean, ax, n_std=2.0, facecolor="none", edgecolor="red", **kwargs
):
    """
    Plots an n-std covariance ellipse for a 2D Gaussian.
    """
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    width, height = 2 * n_std * np.sqrt(vals)
    ellipse = Ellipse(
        xy=mean,
        width=width,
        height=height,
        angle=theta,
        facecolor=facecolor,
        edgecolor=edgecolor,
        **kwargs,
    )
    ax.add_patch(ellipse)


if __name__ == "__main__":

    # samples = sample_skew_laplace(
    #     mu=np.zeros((1, 1)), sigma=np.sqrt(np.array([[0.05]])), lam=0.1, n_samples=10000
    # )
    # Plotting parameters
    plt.rc("text", usetex=True)
    plt.rc("font", family="serif", size=14)
    plt.rc("lines", linewidth=2)
    plt.rc("axes", grid=True)
    plt.rc("grid", linestyle="--")
    R = np.array([[1e-2]])
    std_dev_true = np.sqrt(R)
    samples = randvec(
        cov=np.square(std_dev_true), num_samples=10000, method="skew_laplace"
    )
    samples = samples.reshape((-1, 1))
    # Plot histogram
    fig, ax = plt.subplots()
    ax.hist(
        samples,
        bins="fd",
        alpha=0.3,
        color="tab:grey",
        edgecolor="black",
        density=True,
    )
    ax.set_xlabel("Measurement Error")
    ax.set_ylabel("Probability")
    mu_sl, std_sl, lambda_sl = fit_skew_laplace(samples)
    x = np.linspace(-1, 4, 500)
    pdf_sl = skew_laplace_pdf(x, mu=mu_sl, sigma=std_sl, lam=lambda_sl)
    pdf_sl_true = skew_laplace_pdf(x, mu=0, sigma=std_dev_true[0, 0], lam=0.1)
    # Fit GMM
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
    ax.plot(
        x,
        pdf_sl_true,
        "-",
        linewidth=2,
        color="tab:green",
        label=f"True Skew-Laplace\nμ={0.00}, σ={std_dev_true[0,0]:.2f}, λ={0.100}",
    )
    ax.plot(
        x,
        pdf_sl,
        "--",
        linewidth=2,
        color="tab:red",
        label=f"Skew-Laplace Fit\nμ={mu_sl:.2f}, σ={std_sl:.2f}, λ={lambda_sl:.3f}",
    )
    # Plot GMM PDF
    ax.plot(
        x,
        pdf_gmm,
        "-.",
        linewidth=2,
        color="tab:purple",
        label=f"Gaussian Mixture Fit\nμ=[{means_str}]\nσ=[{stds_str}]\nw=[{weights_str}]",
    )
    print(gmm_means)
    print(gmm_stds)
    print(gmm_weights)
    # Generating presentation figure
    # x = np.linspace(-1, 4, 500)
    # pdf_sl = skew_laplace_pdf(x, mu=-0.05, sigma=0.3, lam=0.6)
    # pdf_gauss = scipy.stats.norm.pdf(x, 0.1, 0.17)
    # pdf_gauss_1 = scipy.stats.norm.pdf(x, 0.5, 0.25)
    # pdf_gauss_2 = scipy.stats.norm.pdf(x, 1, 0.35)
    # pdf_gauss_3 = scipy.stats.norm.pdf(x, 1.5, 0.55)

    # ax.plot(
    #     x,
    #     pdf_gauss,
    #     "--",
    #     linewidth=2,
    #     color="tab:green",
    # )
    # ax.plot(
    #     x,
    #     pdf_gauss_1,
    #     "--",
    #     linewidth=2,
    #     color="tab:red",
    # )
    # ax.plot(
    #     x,
    #     pdf_gauss_2,
    #     "--",
    #     linewidth=2,
    #     color="tab:red",
    # )
    # ax.plot(
    #     x,
    #     pdf_gauss_3,
    #     "--",
    #     linewidth=2,
    #     color="tab:red",
    # )
    # sl_amp = 3.5
    # plt.fill_between(
    #     x, pdf_sl * sl_amp, color="tab:blue", alpha=0.2, label="Skewed Covering Density"
    # )
    # ax.plot(
    #     x,
    #     pdf_sl * sl_amp,
    #     "-",
    #     linewidth=2,
    #     color="tab:blue",
    # )
    ax.legend(fontsize=10, loc="upper right", frameon=True)
    fig.tight_layout()
    plt.show()

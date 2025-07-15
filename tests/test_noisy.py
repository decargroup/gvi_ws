# %%
import numpy as np
import scipy.linalg
import navlie as nav
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
from gvi_ws.util.psd import (
    force_sym_PSD,
    force_sym,
)
from navlie.lib.models import BodyFrameVelocity, LinearMeasurement

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
    beta = np.random.gamma(shape=1, scale=2, size=n_samples) # ~ Gamma(1,2)
    z = np.random.normal(loc=mu + beta * lam, scale=np.sqrt(beta) * sigma)
    return z

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


def plot_cov_ellipse(cov, mean, ax, n_std=2.0, facecolor='none', edgecolor='red', **kwargs):
    """
    Plots an n-std covariance ellipse for a 2D Gaussian.
    """
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    width, height = 2 * n_std * np.sqrt(vals)
    ellipse = Ellipse(xy=mean, width=width, height=height, angle=theta,
                      facecolor=facecolor, edgecolor=edgecolor, **kwargs)
    ax.add_patch(ellipse)
if __name__=="__main__":
    mu = np.array([0.0, 0.0])
    cov = np.array([[1.0, 0.8],
                    [0.8, 1.0]])
    lam = np.array([2.0, -1.0])

    # Sample and compute empirical mean and covariance
    samples = sample_skew_laplace_mv(mu, cov, lam, num_samples=10000)
    emp_mean = np.mean(samples, axis=1)
    emp_cov = np.cov(samples)

    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(samples[0], samples[1], alpha=0.3, s=5, color="purple", label="Samples")

    # Draw 1-sigma and 2-sigma ellipses
    plot_cov_ellipse(emp_cov, emp_mean, ax, n_std=1.0, edgecolor='blue', label="1σ ellipse")
    plot_cov_ellipse(emp_cov, emp_mean, ax, n_std=2.0, edgecolor='red', label="2σ ellipse")

    ax.set_xlabel("Dimension 1")
    ax.set_ylabel("Dimension 2")
    ax.set_title("Multivariate Skew-Laplace Samples with Covariance Ellipses")
    ax.grid(True)
    ax.axis("equal")
    ax.legend()
    plt.show()

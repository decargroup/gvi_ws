# %%
import numpy as np
import scipy as sp
import navlie as nav
import matplotlib.pyplot as plt

from math import factorial
import itertools
from numpy.polynomial.hermite_e import hermeroots
from scipy.stats.distributions import chi2
from scipy.special import eval_hermitenorm
from navlie.filters import SigmaPointKalmanFilter
from navlie.filters import generate_sigmapoints
from matplotlib.patches import Ellipse
# %%

def compute_weight(order_p, sigma_points:np.ndarray):
    H_p = compute_hermite(order_p-1)
    weights = []
    for sigma_i in sigma_points:
        num = factorial(order_p)
        den = order_p*eval_hermite(H_p, sigma_i)
        w_i = num / den**2
        weights.append(w_i)
    return weights

def compute_hermite_coeffs(order_p) -> np.ndarray:
    if order_p == 0:
        return np.array([1])
    elif order_p == 1:
        return np.array([1, 0])
    else:
        h_p = np.append(compute_hermite_coeffs(order_p-1),0)
        h_p_2 = -1*(order_p-1)*compute_hermite_coeffs(order_p-2)
        h_p_1 = np.concatenate([[0, 0], h_p_2])
        return h_p + h_p_1

def compute_hermite(order_p):
    poly = compute_hermite_coeffs(order_p)
    return poly

def eval_hermite(polynomial, point):
    weight = 0
    poly = polynomial[::-1]
    for i in range(len(poly)):
        w_i = poly[i]*(point**i)
        weight+=w_i
    return weight

def gh_cubature_nav(p, dof):
    c = np.zeros(p + 1)
    c[-1] = 1
    sigma_points_scalar = hermeroots(c)
    weights_scalar = np.zeros(p)
    for i in range(p):
        weights_scalar[i] = factorial(p) / (
            (p * eval_hermitenorm(p - 1, sigma_points_scalar[i])) ** 2
        )

    # Generate all p^dof collections of indexes by
    # transforming numbers 0...p^dof-1) into p-base system
    # and by adding 1 to each digit
    num = np.linspace(0, p ** (dof) - 1, p ** (dof))
    ind = np.zeros((dof, p**dof))
    for i in range(dof):
        ind[i, :] = num % p
        num = num // p

    sigma_points = np.zeros((dof, p**dof))
    w = np.zeros(p**dof)
    for i in range(p**dof):
        w[i] = 1
        sigma_point = []
        for j in range(dof):
            w[i] *= weights_scalar[int(ind[j, i])]
            sigma_point.append(sigma_points_scalar[int(ind[j, i])])
        sigma_points[:, i] = np.vstack(sigma_point).ravel()

    return sigma_points, w

def gh_cubature(order_p, state_dof):
    hermite_poly = compute_hermite(order_p)
    unit_sigma_points = np.roots(hermite_poly)
    unit_sigma_points_stack = np.zeros((order_p**state_dof, state_dof))
    # Form cartesian product of sigma points
    count = 0
    for x in itertools.product(unit_sigma_points, repeat=state_dof):
            unit_sigma_points_stack[count, :] = np.array(x)
            count +=1
    # Form weights from multiplying cartesian product of sigma point weights together
    weights = compute_weight(order_p, unit_sigma_points)
    weights_stack = np.zeros(order_p**state_dof)
    count = 0
    for ws in itertools.product(weights, repeat=state_dof):
        weights_stack[count] = np.prod(ws)
        count += 1
    return unit_sigma_points_stack, weights_stack

def spherical_cubature_nav(dof):
    sigma_points = np.sqrt(dof) * np.block([[np.eye(dof), -np.eye(dof)]])
    w = 1 / (2 * dof) * np.ones((2 * dof))
    return sigma_points, w

def compute_mean(sigma_points, weights, g_x:callable, dof):
    mean = np.zeros((dof, 1))
    for i, w in enumerate(weights):
        mean += g_x(sigma_points[i]) * w
    return mean

def compute_var(sigma_points, weights, g_x:callable, dof, mean_prime):
    variance = np.zeros((dof, dof))
    for i, w in enumerate(weights):
        variance += w * (g_x(sigma_points[i]) - mean_prime) @ (g_x(sigma_points[i]) - mean_prime).T 
    return variance

def produce_distribution(mean, variance, num_samples):
    samples = []
    # dev = np.sqrt(variance)
    for _ in range(num_samples):
        # d = np.random.normal(loc=mean, scale=dev)
        d = np.random.multivariate_normal(mean=mean, cov=variance)
        samples.append(d)

    samples = np.array(samples).reshape((num_samples, -1))
    return samples

def plot_samples_ellipse(mean, variance, samples, linestyle = '-', edgecolor='black', fig = None, ax = None, label=None):
    # variance = np.diag(variance)
    evals, evecs = np.linalg.eig(variance)
    bound = chi2.ppf(0.95,2)
    width = np.sqrt(bound*evals[0])
    height = np.sqrt(bound*evals[1])
    if variance[0,1] == 0:
        if variance[0,0] >= variance[1,1]:
            theta = 0
        else:
            theta = np.pi / 2
    else:
        theta = np.arctan2(evals[0] - variance[0,0], variance[0,1])
    xy = tuple((mean[0], mean[1]))
    ells = Ellipse(xy, width = 2*width, height=2*height, angle=np.rad2deg(theta), edgecolor=edgecolor, linestyle=linestyle, linewidth=2, label=label)
    
    ells.set_fill(False)
    if ax is None:
        fig, ax = plt.subplots()
    
    if samples is not None:
        ax.scatter(samples[:,0], samples[:,1], s=0.2)
    ax.scatter(mean[0], mean[1], s = 15, marker='x', color=edgecolor)    
    
    ax.add_artist(ells)
    return fig, ax


# %%
if __name__=='__main__':

    np.random.seed(11)
    mean = np.array([2,4])
    variance = np.array([[0.8,0.1], 
                        [0.1,0.5]])
    g_x = np.sin
    num_samples = 100000
    samples = produce_distribution(mean, variance, num_samples)
    fig, ax = plot_samples_ellipse(mean, variance, samples)
    plt.savefig("/home/astirl/Documents/courses/notes/mech_642/figs/gauss_distrib.pdf")

    # %% Navlie implementation
    # Transformation g(x) = x^2 i.e order = 2
    # H_p satisfies 2 = 2p - 1
    pth_order = 5
    dof = mean.shape[0]

    unit_sigma_points, weights = gh_cubature_nav(pth_order, dof)
    sqrt_P = np.linalg.cholesky(variance)
    sigma_points = [mean.reshape((-1,1)) + sqrt_P @ sp_i.reshape((-1,1)) 
                    for sp_i in unit_sigma_points.T]
    mean_prime = compute_mean(sigma_points, weights, g_x, dof)

    variance_prime = compute_var(sigma_points, weights, g_x, dof, mean_prime)
    # %%
    # My implementation
    unit_sigma_points, weights = gh_cubature(pth_order, dof)

    sigma_points = [mean.reshape((-1,1)) + sqrt_P @ sp_i.reshape((-1,1)) 
                    for sp_i in unit_sigma_points]

    mean_prime = compute_mean(sigma_points, weights, g_x, dof)

    variance_prime = compute_var(sigma_points, weights, g_x, dof, mean_prime)

    # %% 
    # Spherical Cubature
    unit_sigma_points_sp, weights = spherical_cubature_nav(dof)
    sigma_points_sp = [mean.reshape((-1,1)) + sqrt_P @ sp_i.reshape((-1,1)) 
                    for sp_i in unit_sigma_points_sp.T]
    mean_prime_sp = compute_mean(sigma_points_sp, weights, g_x, dof)
    variance_prime_sp = compute_var(sigma_points_sp, weights, g_x, dof, mean_prime_sp)

    # %%
    samples_trans = g_x(samples)
    lb = f"Gauss-Hermite (order {pth_order})"
    fig,ax = plot_samples_ellipse(mean_prime, variance_prime, samples_trans, edgecolor='red', linestyle='--', label=lb)
    fig,ax = plot_samples_ellipse(mean_prime_sp, variance_prime_sp, samples=None, linestyle='--', edgecolor='green', fig=fig, ax=ax, label="Spherical")
    mean_actual = np.mean(samples_trans, axis=0).reshape((-1,1))
    variance_actual = np.cov(samples_trans[:,0], samples_trans[:,1])
    fig,ax = plot_samples_ellipse(mean_actual, variance_actual, samples=None,  edgecolor='black', fig=fig, ax=ax, label="Actual")
    ax.legend()
    # plt.savefig(f"/home/astirl/Documents/courses/notes/mech_642/figs/trans_distrib_{pth_order}_sin.pdf")

    # %%

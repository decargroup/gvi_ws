import numpy as np
from scipy.optimize import minimize

def skew_laplace_logpdf(z, mu, sigma, lam):
    alpha = np.sqrt(1 + (lam ** 2 ) / (sigma ** 2))
    u = (z - mu) / sigma
    logpdf = np.log(1 / (2 *sigma * alpha)) + lam * u / sigma - alpha * np.abs(u)
    return logpdf

def neg_log_likelihood(params, data):
    mu, log_sigma, lam = params
    sigma = np.exp(log_sigma) # Force positive sigma
    logpdf_vals = skew_laplace_logpdf(data, mu, sigma, lam)
    return -np.sum(logpdf_vals)

def fit_skew_laplace(data, init_params = None):
    """
    Fit the skew-Laplace parameters (mu, sigma, lambda) to data via MLE.
    """
    if init_params is None:
       mu_init = np.median(data)
       sigma_init = np.std(data)
       lam_init = 0.0 # Assume symmetric to start.
       init_params = [mu_init, np.log(sigma_init), lam_init]
    
    result = minimize(neg_log_likelihood, init_params, args = (data, ), method = "L-BFGS-B")
    mu_hat, log_sigma_hat, lam_hat = result.x
    return mu_hat, np.exp(log_sigma_hat), lam_hat

def skew_laplace_pdf(x, mu, sigma, lam):
    """
    Univariate skew-Laplace PDF.
    
    Parameters:
        x     : Input array of values.
        mu    : Location parameter.
        sigma : Scale (> 0).
        lam   : Skewness parameter.
    
    Returns:
        PDF values at each point in x.
    """
    alpha = np.sqrt(1 + (lam**2) / (sigma**2))
    u = (x - mu) / sigma
    pdf = (1 / (2 * sigma * alpha)) * np.exp(lam * u / sigma - alpha * np.abs(u))
    return pdf


import numpy as np
from scipy.optimize import minimize


def skew_laplace_logpdf(z, mu, sigma, lam):
    alpha = np.sqrt(1 + (lam**2) / (sigma**2))
    u = (z - mu) / sigma
    logpdf = np.log(1 / (2 * sigma * alpha)) + lam * u / sigma - alpha * np.abs(u)
    return logpdf


def neg_log_likelihood(params, data):
    mu, log_sigma, lam = params
    sigma = np.exp(log_sigma)  # Force positive sigma
    logpdf_vals = skew_laplace_logpdf(data, mu, sigma, lam)
    return -np.sum(logpdf_vals)


def fit_skew_laplace(data, init_params=None):
    """
    Fit the skew-Laplace parameters (mu, sigma, lambda) to data via MLE.
    """
    if init_params is None:
        mu_init = np.median(data)
        sigma_init = np.std(data)
        lam_init = 0.0  # Assume symmetric to start.
        init_params = [mu_init, np.log(sigma_init), lam_init]

    result = minimize(neg_log_likelihood, init_params, args=(data,), method="L-BFGS-B")
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


def fit_skew_laplace_fixed_mu(data, init_params=None):
    """
    Fit skew-Laplace parameters (sigma, lambda) to data via MLE,
    with mu fixed to zero.
    """
    # Initialization
    if init_params is None:
        sigma_init = np.std(data)
        lam_init = 0.0  # Assume symmetric to start
        init_params = [np.log(sigma_init), lam_init]  # only sigma, lambda

    # Objective function that assumes mu = 0
    def neg_log_likelihood_fixed(params, data):
        log_sigma, lam = params
        sigma = np.exp(log_sigma)
        mu = 0.0  # fixed
        # Make sure your original neg_log_likelihood takes these arguments
        return neg_log_likelihood([mu, log_sigma, lam], data)

    # Optimize only sigma, lambda
    result = minimize(
        neg_log_likelihood_fixed, init_params, args=(data,), method="L-BFGS-B"
    )

    log_sigma_hat, lam_hat = result.x
    mu_hat = 0.0
    return mu_hat, np.exp(log_sigma_hat), lam_hat


def two_piece_cauchy_logpdf(z, mu, c_pos, c_neg):
    """
    Log-PDF of the normalized two-piece (asymmetric) Cauchy distribution.

    Parameters
    ----------
    z      : array-like
    mu     : location (float)
    c_pos  : scale for x >= mu (float, >0)
    c_neg  : scale for x < mu (float, >0)

    Returns
    -------
    logpdf : ndarray of same shape as z
    """
    z = np.asarray(z)
    logpdf = np.empty_like(z, dtype=float)

    # Normalizing constant A = 2 / (pi * (c_pos + c_neg))
    # logA = log(2) - log(pi*(c_pos + c_neg))
    logA = np.log(2.0) - (np.log(np.pi) + np.log(c_pos + c_neg))

    pos_mask = z >= mu
    neg_mask = ~pos_mask

    # use log1p for stability: log(1 + ((z-mu)/c)^2) = log1p(((z-mu)/c)**2)
    if np.any(pos_mask):
        u = (z[pos_mask] - mu) / c_pos
        logpdf[pos_mask] = logA - np.log1p(u * u)

    if np.any(neg_mask):
        u = (z[neg_mask] - mu) / c_neg
        logpdf[neg_mask] = logA - np.log1p(u * u)

    return logpdf


def two_piece_cauchy_pdf(x, mu, c_pos, c_neg):
    """
    PDF corresponding to two_piece_cauchy_logpdf.
    """
    return np.exp(two_piece_cauchy_logpdf(x, mu, c_pos, c_neg))


def two_piece_cauchy_negloglik(params, data):
    """
    Negative log-likelihood to minimize.

    params: [mu, log_c_pos, log_c_neg] (we optimize logs to enforce positivity)
    """
    mu, log_c_pos, log_c_neg = params
    c_pos = np.exp(log_c_pos)
    c_neg = np.exp(log_c_neg)

    # If extremely small scales, return large value to deter optimizer
    if c_pos <= 0 or c_neg <= 0:
        return np.inf

    logpdf_vals = two_piece_cauchy_logpdf(data, mu, c_pos, c_neg)
    # return negative sum of logpdf
    return -np.sum(logpdf_vals)


def fit_two_piece_cauchy(data, init_params=None, method="L-BFGS-B"):
    """
    Fit mu, c_pos, c_neg by MLE.

    Returns (mu_hat, c_pos_hat, c_neg_hat)
    """
    data = np.asarray(data)
    if init_params is None:
        mu_init = np.median(data)
        # Use robust scale estimate for initialization (MAD or std clipped)
        sigma_init = np.std(data) if np.std(data) > 0 else 1.0
        init_params = [mu_init, np.log(sigma_init), np.log(sigma_init)]

    result = minimize(
        two_piece_cauchy_negloglik,
        x0=np.asarray(init_params),
        args=(data,),
        method=method,
        bounds=[(None, None), (np.log(1e-8), None), (np.log(1e-8), None)],
    )
    mu_hat, log_c_pos_hat, log_c_neg_hat = result.x
    return mu_hat, float(np.exp(log_c_pos_hat)), float(np.exp(log_c_neg_hat))

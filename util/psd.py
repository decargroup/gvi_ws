from numpy import linalg as la
import numpy as np


def nearestPD(A):
    """Find the nearest positive-definite matrix to input

    A Python/Numpy port of John D'Errico's `nearestSPD` MATLAB code [1], which
    credits [2].

    [1] https://www.mathworks.com/matlabcentral/fileexchange/42885-nearestspd

    [2] N.J. Higham, "Computing a nearest symmetric positive semidefinite
    matrix" (1988): https://doi.org/10.1016/0024-3795(88)90223-6
    """

    B = (A + A.T) / 2
    _, s, V = la.svd(B)

    H = np.dot(V.T, np.dot(np.diag(s), V))

    A2 = (B + H) / 2

    A3 = (A2 + A2.T) / 2

    if isPD(A3):
        return A3

    spacing = np.spacing(la.norm(A))
    # The above is different from [1]. It appears that MATLAB's `chol` Cholesky
    # decomposition will accept matrixes with exactly 0-eigenvalue, whereas
    # Numpy's will not. So where [1] uses `eps(mineig)` (where `eps` is Matlab
    # for `np.spacing`), we use the above definition. CAVEAT: our `spacing`
    # will be much larger than [1]'s `eps(mineig)`, since `mineig` is usually on
    # the order of 1e-16, and `eps(1e-16)` is on the order of 1e-34, whereas
    # `spacing` will, for Gaussian random matrixes of small dimension, be on
    # othe order of 1e-16. In practice, both ways converge, as the unit test
    # below suggests.
    I = np.eye(A.shape[0])
    k = 1
    while not isPD(A3):
        mineig = np.min(np.real(la.eigvals(A3)))
        A3 += I * (-mineig * k**2 + spacing)
        k += 1

    return A3


def isPD(B):
    """Returns true when input is positive-definite, via Cholesky"""
    try:
        _ = la.cholesky(B)
        return True
    except la.LinAlgError:
        return False
    
def regularize(A, cond_threshold=1e12, epsilon=1e-6):
    """
    Regularizes a matrix if its condition number is too high.

    Parameters:
        A (np.ndarray): The input matrix.
        cond_threshold (float): Threshold for condition number beyond which regularization is applied.
        epsilon (float): Small value to add to the diagonal for regularization.

    Returns:
        np.ndarray: The regularized matrix.
    """
    cond_number = la.cond(A)
    if cond_number > cond_threshold or not np.isfinite(cond_number):
        # print(f"Condition number is too high ({cond_number:.2e}). Regularizing the matrix.")
        A += epsilon * np.eye(A.shape[0])  # Add small diagonal regularization
    return A
    
def force_PSD(A, cond_threshold=1e12, epsilon=1e-6):
    
    A = regularize(A, cond_threshold=cond_threshold, epsilon=epsilon)
    if not isPD(A):
        return nearestPD(A)
    else:
        return A
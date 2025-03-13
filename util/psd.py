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


def isPD(B:np.ndarray) -> bool:
    """Returns true when input is positive-definite, via Cholesky"""
    try:
        _ = la.cholesky(B)
        return True
    except la.LinAlgError:
        return False
    
def regularize(A: np.ndarray, cond_threshold=1e6, eps_min=1e-6, eps_max=1e3) -> np.ndarray:
    """ Regularize matrix by adaptively adjusting the diagonal perturbation. """
    cond_number = np.linalg.cond(A)
    if cond_number > cond_threshold or not np.isfinite(cond_number):
        eps = eps_min
        while cond_number > cond_threshold and eps < eps_max:
            A += eps * np.eye(A.shape[0])
            cond_number = np.linalg.cond(A)
            eps *= 2  # Gradually increase epsilon
        print(f"Regularized condition number {cond_number}, with epsilon = {eps}")
    return A
    
def force_PSD(A:np.ndarray) -> np.ndarray:
    
    # A = regularize(A, cond_threshold=cond_threshold, epsilon=epsilon)
    if not isPD(A):
        return nearestPD(A)
    else:
        return A
    
def force_sym(A):
    A = (A + A.T) / 2
    return A

# def nearest_PSD(A:np.ndarray, epsilon:float = 1e-8) -> np.ndarray:
#     A = force_sym(A)

#     # Perform eigendecomposition: Σ = V D V^T
#     eigvals, eigvecs = np.linalg.eigh(A)  # Eigen decomposition for symmetric matrices
    
#     # Modify D to D+ (element-wise max with 0)
#     D_plus = np.diag(np.maximum(eigvals, 0))
    
#     # Reconstruct Σ+ = V D+ V^T + epsilon * I
#     while True:
#         A = eigvecs @ D_plus @ eigvecs.T + epsilon * np.eye(A.shape[0])
#         if isPD(A):
#             break
#         else:
#             epsilon *= 10
    
#     return A
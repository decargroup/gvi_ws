import numpy as np


def force_block_banded_sparsity(
    info_matrix: np.ndarray, block_size: int = 3
) -> np.ndarray:
    info_matrix = info_matrix.copy()
    n = info_matrix.shape[0]
    num_blocks = n // block_size

    assert n % block_size == 0, "Matrix size must be divisible by block size"

    sparse_matrix = np.zeros_like(info_matrix)

    for i in range(num_blocks):
        for j in range(num_blocks):
            if abs(i - j) <= 1:
                row_start = i * block_size
                row_end = (i + 1) * block_size
                col_start = j * block_size
                col_end = (j + 1) * block_size

                sparse_matrix[row_start:row_end, col_start:col_end] = info_matrix[
                    row_start:row_end, col_start:col_end
                ]

    return sparse_matrix


def compute_sparse_inverse(L, D):
    S = np.zeros_like(D)
    n = D.shape[0]
    # Iterate backwards over rows
    for j in range(n - 1, -1, -1):  # k = K, K-1, ..., 0
        # Diagonal element
        S[j, j] = 1 / D[j, j]
        # Iterate across columns
        for k in range(j, -1, -1):  # j = k, k-1, ..., 0
            for ell in range(k + 1, n):
                S[j, k] -= S[j, ell] * L[ell, k]
            S[k, j] = S[j, k]  # Symmetric matrix
    return S


# @jax.jit
# def jax_compute_sparse_inverse(L, D):
#     n = D.shape[0]
#     S = jnp.zeros_like(D)

#     def outer_loop_body(j, S):
#         S = S.at[j, j].set(1.0 / D[j, j])

#         def inner_loop_body(k, S):
#             def update(S):
#                 def ell_body(ell, val):
#                     return val - S[j, ell] * L[ell, k]

#                 S_jk = lax.fori_loop(k + 1, n, ell_body, S[j, k])
#                 S = S.at[j, k].set(S_jk)
#                 S = S.at[k, j].set(S_jk)  # Symmetry
#                 return S

#             return lax.cond(j >= k, update, lambda x: x, S)

#         S = lax.fori_loop(0, j + 1, inner_loop_body, S)
#         return S

#     S = lax.fori_loop(0, n, lambda i, S: outer_loop_body(n - 1 - i, S), S)
#     return S

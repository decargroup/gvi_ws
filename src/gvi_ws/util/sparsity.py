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

from __future__ import annotations

import numpy as np


def _as_square_matrix(matrix: np.ndarray | list[list[float]]) -> np.ndarray:
    array = np.asarray(matrix, dtype=float)
    if array.ndim != 2 or array.shape[0] != array.shape[1]:
        raise ValueError(f"Expected a square matrix, got shape {array.shape}.")
    return array


def _as_eigenfunction_stack(
    phi_values: np.ndarray | list[float] | list[list[float]],
    dimension: int,
) -> np.ndarray:
    array = np.asarray(phi_values, dtype=float)
    if array.shape[-1] != dimension:
        raise ValueError(
            "Expected the last axis of the eigenfunction array to match the number "
            f"of eigenvalues ({dimension}), got shape {array.shape}."
        )
    return array


def koopman_quadratic_observable(
    phi_values: np.ndarray | list[float] | list[list[float]],
    matrix_p: np.ndarray | list[list[float]],
) -> np.ndarray:
    """Evaluate V(x) = sum_{i,j} P_ij phi_i(x) phi_j(x)."""
    matrix = _as_square_matrix(matrix_p)
    phi = _as_eigenfunction_stack(phi_values, matrix.shape[0])
    return np.einsum("...i,ij,...j->...", phi, matrix, phi)


def koopman_quadratic_time_derivative(
    phi_values: np.ndarray | list[float] | list[list[float]],
    matrix_p: np.ndarray | list[list[float]],
    eigenvalues: np.ndarray | list[float],
) -> np.ndarray:
    """Evaluate dot V(x) from the Koopman eigenvalue identity.

    This implements
        dot V(x) = sum_{i,j} P_ij (lambda_i + lambda_j) phi_i(x) phi_j(x).
    """
    matrix = _as_square_matrix(matrix_p)
    lambda_array = np.asarray(eigenvalues, dtype=float)
    if lambda_array.ndim != 1 or lambda_array.shape[0] != matrix.shape[0]:
        raise ValueError(
            "Expected one eigenvalue per row/column of P, "
            f"got shape {lambda_array.shape} for P with shape {matrix.shape}."
        )
    phi = _as_eigenfunction_stack(phi_values, matrix.shape[0])
    lambda_sum = lambda_array[:, None] + lambda_array[None, :]
    return np.einsum("...i,ij,...j->...", phi, matrix * lambda_sum, phi)

from __future__ import annotations

import numpy as np


def as_points(points: np.ndarray | list[list[float]] | list[float]) -> np.ndarray:
    """Return points as a 2D float array with shape (n_points, dimension)."""
    array = np.asarray(points, dtype=float)
    if array.ndim == 1:
        array = array[None, :]
    if array.ndim != 2:
        raise ValueError(f"Expected a 1D or 2D array of points, got shape {array.shape}.")
    return array


def as_sigmas(sigmas: float | np.ndarray | list[float], dimension: int) -> np.ndarray:
    """Broadcast scalar or vector kernel widths to one width per dimension."""
    sigma_array = np.asarray(sigmas, dtype=float)
    if sigma_array.ndim == 0:
        sigma_array = np.repeat(float(sigma_array), dimension)
    if sigma_array.shape != (dimension,):
        raise ValueError(
            f"Expected {dimension} kernel widths, got shape {sigma_array.shape}."
        )
    if np.any(sigma_array <= 0.0):
        raise ValueError("Kernel widths must be strictly positive.")
    return sigma_array


def gaussian_kernel_matrix(
    x_points: np.ndarray | list[list[float]] | list[float],
    y_points: np.ndarray | list[list[float]] | list[float],
    sigmas: float | np.ndarray | list[float],
) -> np.ndarray:
    """Evaluate an anisotropic Gaussian kernel on two point clouds."""
    x_array = as_points(x_points)
    y_array = as_points(y_points)
    if x_array.shape[1] != y_array.shape[1]:
        raise ValueError("Point clouds must have the same ambient dimension.")

    sigma_array = as_sigmas(sigmas, x_array.shape[1])
    exponent = np.zeros((len(x_array), len(y_array)), dtype=float)
    for axis, sigma in enumerate(sigma_array):
        difference = x_array[:, None, axis] - y_array[None, :, axis]
        exponent += (difference / sigma) ** 2
    return np.exp(-0.5 * exponent)


def gaussian_grad_x(
    x_points: np.ndarray | list[list[float]] | list[float],
    y_points: np.ndarray | list[list[float]] | list[float],
    sigmas: float | np.ndarray | list[float],
    axis: int,
    base_kernel: np.ndarray | None = None,
) -> np.ndarray:
    """Differentiate the Gaussian kernel with respect to the first argument."""
    x_array = as_points(x_points)
    y_array = as_points(y_points)
    sigma_array = as_sigmas(sigmas, x_array.shape[1])
    kernel_values = (
        gaussian_kernel_matrix(x_array, y_array, sigma_array)
        if base_kernel is None
        else base_kernel
    )
    difference = x_array[:, None, axis] - y_array[None, :, axis]
    return -(difference / sigma_array[axis] ** 2) * kernel_values


def gaussian_grad_y(
    x_points: np.ndarray | list[list[float]] | list[float],
    y_points: np.ndarray | list[list[float]] | list[float],
    sigmas: float | np.ndarray | list[float],
    axis: int,
    base_kernel: np.ndarray | None = None,
) -> np.ndarray:
    """Differentiate the Gaussian kernel with respect to the second argument."""
    x_array = as_points(x_points)
    y_array = as_points(y_points)
    sigma_array = as_sigmas(sigmas, x_array.shape[1])
    kernel_values = (
        gaussian_kernel_matrix(x_array, y_array, sigma_array)
        if base_kernel is None
        else base_kernel
    )
    difference = x_array[:, None, axis] - y_array[None, :, axis]
    return (difference / sigma_array[axis] ** 2) * kernel_values


def gaussian_mixed_derivative_xy(
    x_points: np.ndarray | list[list[float]] | list[float],
    y_points: np.ndarray | list[list[float]] | list[float],
    sigmas: float | np.ndarray | list[float],
    axis_x: int,
    axis_y: int,
    base_kernel: np.ndarray | None = None,
) -> np.ndarray:
    """Evaluate the mixed derivative d^2 K / (dx_axis_x dy_axis_y)."""
    x_array = as_points(x_points)
    y_array = as_points(y_points)
    sigma_array = as_sigmas(sigmas, x_array.shape[1])
    kernel_values = (
        gaussian_kernel_matrix(x_array, y_array, sigma_array)
        if base_kernel is None
        else base_kernel
    )

    difference_x = x_array[:, None, axis_x] - y_array[None, :, axis_x]
    difference_y = x_array[:, None, axis_y] - y_array[None, :, axis_y]
    diagonal_term = 1.0 / sigma_array[axis_x] ** 2 if axis_x == axis_y else 0.0
    product_term = (
        difference_x / sigma_array[axis_x] ** 2
    ) * (
        difference_y / sigma_array[axis_y] ** 2
    )
    return (diagonal_term - product_term) * kernel_values

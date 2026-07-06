from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy import linalg

from .kernels import (
    as_points,
    as_sigmas,
    gaussian_grad_x,
    gaussian_grad_y,
    gaussian_kernel_matrix,
    gaussian_mixed_derivative_xy,
)


VectorField = Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True)
class CollocationProblem:
    """Kernel collocation problem for one Koopman eigenfunction."""

    vector_field: VectorField
    linearization: np.ndarray
    collocation_points: np.ndarray
    eigenvalue: float
    sigmas: np.ndarray
    linear_part: np.ndarray
    nugget: float = 1e-10
    label: str = ""

    @property
    def dimension(self) -> int:
        return int(self.collocation_points.shape[1])

    @property
    def n_constraints(self) -> int:
        return int(1 + self.dimension + len(self.collocation_points))


@dataclass
class CollocationModel:
    """Fitted representer for the nonlinear correction h(x)."""

    problem: CollocationProblem
    coefficients: np.ndarray
    flow_values: np.ndarray
    targets: np.ndarray
    gram_matrix: np.ndarray

    def nonlinear_part(self, points: np.ndarray | list[list[float]] | list[float]) -> np.ndarray:
        point_array = as_points(points)
        features = representer_matrix(
            point_array,
            self.problem.collocation_points,
            self.flow_values,
            self.problem.sigmas,
            self.problem.eigenvalue,
        )
        return features @ self.coefficients

    def eigenfunction(self, points: np.ndarray | list[list[float]] | list[float]) -> np.ndarray:
        point_array = as_points(points)
        return self.nonlinear_part(point_array) + point_array @ self.problem.linear_part


def normalize_vector(vector: np.ndarray) -> np.ndarray:
    """Normalize a vector to a deterministic max-abs scale and orientation."""
    array = np.real_if_close(np.asarray(vector, dtype=complex), tol=1_000)
    if np.iscomplexobj(array):
        raise ValueError("Expected a real eigenvector after removing small imaginary parts.")
    array = np.asarray(array, dtype=float)
    scale = np.max(np.abs(array))
    if scale == 0.0:
        raise ValueError("Cannot normalize the zero vector.")
    array = array / scale
    pivot = int(np.argmax(np.abs(array)))
    if array[pivot] < 0.0:
        array = -array
    return array


def real_vector(vector: np.ndarray) -> np.ndarray:
    """Convert a nearly-real vector to float without changing its scale."""
    array = np.real_if_close(np.asarray(vector, dtype=complex), tol=1_000)
    if np.iscomplexobj(array):
        raise ValueError("Expected a real vector after removing small imaginary parts.")
    return np.asarray(array, dtype=float)


def match_left_eigenvector(linearization: np.ndarray, target_eigenvalue: float) -> np.ndarray:
    """Select the left eigenvector whose eigenvalue matches the requested target."""
    eigenvalues, left_vectors = linalg.eig(linearization, left=True, right=False)
    index = int(np.argmin(np.abs(eigenvalues - target_eigenvalue)))
    chosen_value = np.real_if_close(eigenvalues[index], tol=1_000)
    if np.iscomplexobj(chosen_value):
        raise ValueError(f"Matched eigenvalue {chosen_value} is not real.")
    return normalize_vector(left_vectors[:, index])


def safe_relative_error(
    prediction: np.ndarray,
    truth: np.ndarray,
    floor: float = 1e-9,
) -> np.ndarray:
    """Compute a stable pointwise relative error that avoids division by zero."""
    truth_array = np.asarray(truth, dtype=float)
    denominator = np.maximum(np.abs(truth_array), floor)
    return np.abs(np.asarray(prediction, dtype=float) - truth_array) / denominator


def validate_linear_part(
    linearization: np.ndarray,
    eigenvalue: float,
    linear_part: np.ndarray,
    tolerance: float = 1e-8,
) -> None:
    """Check that w^T E = lambda w^T for the provided linear part."""
    residual = linear_part @ linearization - eigenvalue * linear_part
    if np.linalg.norm(residual, ord=np.inf) > tolerance:
        raise ValueError(
            "The supplied linear part is not a left eigenvector of the linearization."
        )


def collocation_targets(
    points: np.ndarray,
    flow_values: np.ndarray,
    linearization: np.ndarray,
    linear_part: np.ndarray,
) -> np.ndarray:
    """Assemble the right-hand side Y from the PDE constraints."""
    dimension = points.shape[1]
    nonlinear_flow = flow_values - points @ linearization.T
    targets = np.zeros(1 + dimension + len(points), dtype=float)
    targets[1 + dimension :] = -(nonlinear_flow @ linear_part)
    return targets


def collocation_features(
    evaluation_points: np.ndarray,
    collocation_points: np.ndarray,
    flow_values: np.ndarray,
    sigmas: np.ndarray,
    eigenvalue: float,
    base_kernel: np.ndarray | None = None,
) -> np.ndarray:
    """Evaluate the collocation functionals against the kernel."""
    kernel_values = (
        gaussian_kernel_matrix(evaluation_points, collocation_points, sigmas)
        if base_kernel is None
        else base_kernel
    )
    features = -eigenvalue * kernel_values
    for axis in range(collocation_points.shape[1]):
        features += flow_values[:, axis][None, :] * gaussian_grad_y(
            evaluation_points,
            collocation_points,
            sigmas,
            axis,
            base_kernel=kernel_values,
        )
    return features


def representer_matrix(
    evaluation_points: np.ndarray,
    collocation_points: np.ndarray,
    flow_values: np.ndarray,
    sigmas: np.ndarray,
    eigenvalue: float,
) -> np.ndarray:
    """Assemble K(x, \u03d5~) for a batch of evaluation points."""
    point_array = as_points(evaluation_points)
    collocation_array = as_points(collocation_points)
    sigma_array = as_sigmas(sigmas, collocation_array.shape[1])

    dimension = collocation_array.shape[1]
    origin = np.zeros((1, dimension), dtype=float)
    features = np.zeros((len(point_array), 1 + dimension + len(collocation_array)), dtype=float)
    features[:, 0] = gaussian_kernel_matrix(point_array, origin, sigma_array)[:, 0]
    for axis in range(dimension):
        features[:, 1 + axis] = gaussian_grad_y(point_array, origin, sigma_array, axis)[:, 0]
    features[:, 1 + dimension :] = collocation_features(
        point_array,
        collocation_array,
        flow_values,
        sigma_array,
        eigenvalue,
    )
    return features


def gram_matrix(
    points: np.ndarray,
    flow_values: np.ndarray,
    sigmas: np.ndarray,
    eigenvalue: float,
) -> np.ndarray:
    """Build K(\u03d5~, \u03d5~) for the kernel-collocation constraints."""
    point_array = as_points(points)
    sigma_array = as_sigmas(sigmas, point_array.shape[1])
    n_points, dimension = point_array.shape
    offset = 1 + dimension
    total_size = offset + n_points

    origin = np.zeros((1, dimension), dtype=float)
    matrix = np.zeros((total_size, total_size), dtype=float)

    matrix[0, 0] = gaussian_kernel_matrix(origin, origin, sigma_array)[0, 0]
    for axis_x in range(dimension):
        matrix[0, 1 + axis_x] = gaussian_grad_y(origin, origin, sigma_array, axis_x)[0, 0]
        matrix[1 + axis_x, 0] = matrix[0, 1 + axis_x]
        for axis_y in range(dimension):
            matrix[1 + axis_x, 1 + axis_y] = gaussian_mixed_derivative_xy(
                origin,
                origin,
                sigma_array,
                axis_x,
                axis_y,
            )[0, 0]

    matrix[0, offset:] = collocation_features(
        origin,
        point_array,
        flow_values,
        sigma_array,
        eigenvalue,
    )[0]
    matrix[offset:, 0] = matrix[0, offset:]

    for axis_x in range(dimension):
        derivative_row = -eigenvalue * gaussian_grad_x(
            origin,
            point_array,
            sigma_array,
            axis_x,
        )[0]
        for axis_y in range(dimension):
            derivative_row += flow_values[:, axis_y] * gaussian_mixed_derivative_xy(
                origin,
                point_array,
                sigma_array,
                axis_x,
                axis_y,
            )[0]
        matrix[1 + axis_x, offset:] = derivative_row
        matrix[offset:, 1 + axis_x] = derivative_row

    base_kernel = gaussian_kernel_matrix(point_array, point_array, sigma_array)
    collocation_block = (eigenvalue**2) * base_kernel
    for axis_x in range(dimension):
        gradient_x = gaussian_grad_x(
            point_array,
            point_array,
            sigma_array,
            axis_x,
            base_kernel=base_kernel,
        )
        gradient_y = -gradient_x
        collocation_block -= eigenvalue * (
            flow_values[:, axis_x][:, None] * gradient_x
            + flow_values[:, axis_x][None, :] * gradient_y
        )
        for axis_y in range(dimension):
            mixed = gaussian_mixed_derivative_xy(
                point_array,
                point_array,
                sigma_array,
                axis_x,
                axis_y,
                base_kernel=base_kernel,
            )
            collocation_block += np.outer(flow_values[:, axis_x], flow_values[:, axis_y]) * mixed

    matrix[offset:, offset:] = collocation_block
    return 0.5 * (matrix + matrix.T)


def fit_collocation_problem(problem: CollocationProblem) -> CollocationModel:
    """Solve the regularized representer system for one eigenfunction."""
    point_array = as_points(problem.collocation_points)
    linearization = np.asarray(problem.linearization, dtype=float)
    sigma_array = as_sigmas(problem.sigmas, point_array.shape[1])
    linear_part = real_vector(problem.linear_part)
    validate_linear_part(linearization, problem.eigenvalue, linear_part)

    flow_values = np.asarray(problem.vector_field(point_array), dtype=float)
    if flow_values.shape != point_array.shape:
        raise ValueError(
            "The vector field must return an array with the same shape as the input points."
        )

    targets = collocation_targets(point_array, flow_values, linearization, linear_part)
    kernel_matrix = gram_matrix(point_array, flow_values, sigma_array, problem.eigenvalue)
    regularized = kernel_matrix + problem.nugget * np.eye(kernel_matrix.shape[0])
    try:
        coefficients = linalg.solve(regularized, targets, assume_a="sym")
    except linalg.LinAlgError:
        coefficients = np.linalg.solve(regularized, targets)

    fitted_problem = CollocationProblem(
        vector_field=problem.vector_field,
        linearization=linearization,
        collocation_points=point_array,
        eigenvalue=float(problem.eigenvalue),
        sigmas=sigma_array,
        linear_part=linear_part,
        nugget=float(problem.nugget),
        label=problem.label,
    )
    return CollocationModel(
        problem=fitted_problem,
        coefficients=coefficients,
        flow_values=flow_values,
        targets=targets,
        gram_matrix=kernel_matrix,
    )

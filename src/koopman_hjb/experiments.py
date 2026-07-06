from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Callable

import numpy as np
from scipy.optimize import fsolve

from .kernels import as_points
from .solver import (
    CollocationModel,
    CollocationProblem,
    fit_collocation_problem,
    match_left_eigenvector,
    safe_relative_error,
)


TrueEigenfunction = Callable[[np.ndarray], np.ndarray]
VectorField = Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True)
class EigenfunctionSpec:
    key: str
    title: str
    eigenvalue: float
    sigmas: tuple[float, ...]
    linear_part: tuple[float, ...] | None = None
    true_eigenfunction: TrueEigenfunction | None = None
    nugget: float = 1e-10


@dataclass(frozen=True)
class ExperimentSpec:
    key: str
    title: str
    section: str
    description: str
    bounds: tuple[tuple[float, float], ...]
    grid_shape: tuple[int, ...]
    linearization: np.ndarray
    vector_field: VectorField
    eigenfunctions: tuple[EigenfunctionSpec, ...]
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def dimension(self) -> int:
        return len(self.bounds)

    def grid_axes(self) -> tuple[np.ndarray, ...]:
        return tuple(
            np.linspace(bound[0], bound[1], count)
            for bound, count in zip(self.bounds, self.grid_shape)
        )

    def mesh(self) -> tuple[np.ndarray, ...]:
        return np.meshgrid(*self.grid_axes(), indexing="ij")

    def grid_points(self) -> np.ndarray:
        return np.stack([axis.ravel() for axis in self.mesh()], axis=1)

    def reshape(self, values: np.ndarray) -> np.ndarray:
        return np.asarray(values, dtype=float).reshape(self.grid_shape)

    def get_eigenfunction(self, key: str) -> EigenfunctionSpec:
        for spec in self.eigenfunctions:
            if spec.key == key:
                return spec
        raise KeyError(f"Unknown eigenfunction '{key}' for experiment '{self.key}'.")

    def linear_part_for(self, spec: EigenfunctionSpec) -> np.ndarray:
        if spec.linear_part is not None:
            return np.asarray(spec.linear_part, dtype=float)
        return match_left_eigenvector(self.linearization, spec.eigenvalue)

    def fit(self, key: str) -> CollocationModel:
        spec = self.get_eigenfunction(key)
        problem = CollocationProblem(
            vector_field=self.vector_field,
            linearization=np.asarray(self.linearization, dtype=float),
            collocation_points=self.grid_points(),
            eigenvalue=spec.eigenvalue,
            sigmas=np.asarray(spec.sigmas, dtype=float),
            linear_part=self.linear_part_for(spec),
            nugget=spec.nugget,
            label=f"{self.key}:{spec.key}",
        )
        return fit_collocation_problem(problem)

    def evaluate_on_grid(self, model: CollocationModel) -> np.ndarray:
        return self.reshape(model.eigenfunction(self.grid_points()))

    def true_on_grid(self, key: str) -> np.ndarray:
        spec = self.get_eigenfunction(key)
        if spec.true_eigenfunction is None:
            raise ValueError(f"Eigenfunction '{key}' does not have a closed-form reference.")
        return self.reshape(spec.true_eigenfunction(self.grid_points()))

    def relative_error_on_grid(self, key: str, learned_values: np.ndarray) -> np.ndarray:
        truth = self.true_on_grid(key)
        return safe_relative_error(np.asarray(learned_values, dtype=float), truth)

    def slice_points(
        self,
        fixed_axis: int,
        fixed_value: float,
        resolution: int = 200,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if self.dimension != 3:
            raise ValueError("Slice evaluation is only available for 3D experiments.")

        varying_axes = [axis for axis in range(self.dimension) if axis != fixed_axis]
        first_axis = np.linspace(*self.bounds[varying_axes[0]], resolution)
        second_axis = np.linspace(*self.bounds[varying_axes[1]], resolution)
        first_mesh, second_mesh = np.meshgrid(first_axis, second_axis, indexing="ij")

        points = np.zeros((resolution * resolution, self.dimension), dtype=float)
        points[:, fixed_axis] = fixed_value
        points[:, varying_axes[0]] = first_mesh.ravel()
        points[:, varying_axes[1]] = second_mesh.ravel()
        return first_axis, second_axis, first_mesh, second_mesh, points

    def evaluate_slice(
        self,
        model: CollocationModel,
        fixed_axis: int,
        fixed_value: float,
        resolution: int = 200,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        first_axis, second_axis, first_mesh, second_mesh, points = self.slice_points(
            fixed_axis=fixed_axis,
            fixed_value=fixed_value,
            resolution=resolution,
        )
        values = model.eigenfunction(points).reshape((resolution, resolution))
        return first_axis, second_axis, first_mesh, second_mesh, values

    def find_equilibria(
        self,
        guesses: tuple[tuple[float, ...], ...],
        tolerance: float = 1e-8,
    ) -> list[np.ndarray]:
        solutions: list[np.ndarray] = []

        def root_function(point: np.ndarray) -> np.ndarray:
            return self.vector_field(as_points(point))[0]

        for guess in guesses:
            root = np.asarray(fsolve(root_function, np.asarray(guess, dtype=float)), dtype=float)
            if not any(np.linalg.norm(root - existing, ord=np.inf) < tolerance for existing in solutions):
                solutions.append(root)
        return solutions

    def describe(self) -> str:
        eigen_lines = ", ".join(
            f"{spec.key}: lambda={spec.eigenvalue:g}, sigma={spec.sigmas}"
            for spec in self.eigenfunctions
        )
        return (
            f"{self.title}\n"
            f"Section: {self.section}\n"
            f"Grid: {self.grid_shape} over {self.bounds}\n"
            f"Eigenfunctions: {eigen_lines}"
        )


def first_analytic_vector_field(points: np.ndarray) -> np.ndarray:
    point_array = as_points(points)
    x1 = point_array[:, 0]
    x2 = point_array[:, 1]
    lambda_1 = -1.0
    lambda_2 = 3.0
    value_1 = (
        -2.0 * lambda_2 * x2 * (x1**2 - x2 - 2.0 * x1 * x2**2 + x2**4)
        + lambda_1 * (x1 + 4.0 * x1**2 * x2 - x2**2 - 8.0 * x1 * x2**3 + 4.0 * x2**5)
    )
    value_2 = (
        2.0 * lambda_1 * (x1 - x2**2) ** 2
        - lambda_2 * (x1**2 - x2 - 2.0 * x1 * x2**2 + x2**4)
    )
    return np.column_stack([value_1, value_2])


def second_analytic_vector_field(points: np.ndarray) -> np.ndarray:
    point_array = as_points(points)
    x1 = point_array[:, 0]
    x2 = point_array[:, 1]
    denominator = 9.0 * x1**2 * x2**2 + 6.0 * x1**2 + 3.0 * x2**2 + np.cos(x2) + 2.0
    value_1 = (
        (7.5 * x2**2 + 5.0) * (x1**3 + x1 + np.sin(x2))
        + (-x1 + x2**3 + 2.0 * x2) * np.cos(x2)
    ) / denominator
    value_2 = (
        2.5 * x1**3
        + 2.5 * x1
        - (3.0 * x1**2 + 1.0) * (-x1 + x2**3 + 2.0 * x2)
        + 2.5 * np.sin(x2)
    ) / denominator
    return np.column_stack([value_1, value_2])


def duffing_vector_field(points: np.ndarray) -> np.ndarray:
    point_array = as_points(points)
    alpha = 1.0
    beta = -1.0
    delta = 0.5
    value_1 = point_array[:, 1]
    value_2 = -delta * point_array[:, 1] - point_array[:, 0] * (
        beta + alpha * point_array[:, 0] ** 2
    )
    return np.column_stack([value_1, value_2])


def gradient_system_vector_field(points: np.ndarray) -> np.ndarray:
    point_array = as_points(points)
    matrix_p = np.array(
        [
            [0.2, 0.1, 0.05],
            [0.1, 0.3, 0.05],
            [0.05, 0.05, 0.2],
        ],
        dtype=float,
    )
    quadratic_part = -2.0 * point_array @ matrix_p.T
    difference = point_array[:, 0] - point_array[:, 1]
    exponential = np.exp(-(difference**2))
    correction = np.column_stack(
        [
            2.0 * difference * exponential,
            -2.0 * difference * exponential,
            np.zeros(len(point_array)),
        ]
    )
    return quadratic_part + correction


def first_analytic_phi_1(points: np.ndarray) -> np.ndarray:
    point_array = as_points(points)
    return point_array[:, 0] - point_array[:, 1] ** 2


def first_analytic_phi_2(points: np.ndarray) -> np.ndarray:
    point_array = as_points(points)
    x1 = point_array[:, 0]
    x2 = point_array[:, 1]
    return -(x1**2) + x2 + 2.0 * x1 * x2**2 - x2**4


def second_analytic_phi_1(points: np.ndarray) -> np.ndarray:
    point_array = as_points(points)
    return point_array[:, 0] - 2.0 * point_array[:, 1] - point_array[:, 1] ** 3


def second_analytic_phi_2(points: np.ndarray) -> np.ndarray:
    point_array = as_points(points)
    return point_array[:, 0] + np.sin(point_array[:, 1]) + point_array[:, 0] ** 3


def build_registry() -> dict[str, ExperimentSpec]:
    return {
        "first_analytic": ExperimentSpec(
            key="first_analytic",
            title="First Analytical Example",
            section="Section 5.1",
            description="Kernel reconstruction of both principal eigenfunctions from the first analytical system.",
            bounds=((-1.0, 1.0), (-1.0, 1.0)),
            grid_shape=(60, 60),
            linearization=np.array([[-1.0, 0.0], [0.0, 3.0]], dtype=float),
            vector_field=first_analytic_vector_field,
            eigenfunctions=(
                EigenfunctionSpec(
                    key="phi_lambda_1",
                    title=r"$\phi_{\lambda_1}$",
                    eigenvalue=-1.0,
                    sigmas=(2.0, 2.0),
                    linear_part=(1.0, 0.0),
                    true_eigenfunction=first_analytic_phi_1,
                    nugget=1e-6,
                ),
                EigenfunctionSpec(
                    key="phi_lambda_2",
                    title=r"$\phi_{\lambda_2}$",
                    eigenvalue=3.0,
                    sigmas=(2.0, 3.0),
                    linear_part=(0.0, 1.0),
                    true_eigenfunction=first_analytic_phi_2,
                    nugget=1e-8,
                ),
            ),
        ),
        "second_analytic": ExperimentSpec(
            key="second_analytic",
            title="Second Analytical Example",
            section="Section 5.2",
            description="Kernel reconstruction of both principal eigenfunctions from the second analytical system.",
            bounds=((1.5, 2.5), (1.5, 2.5)),
            grid_shape=(50, 50),
            linearization=np.array([[4.0 / 3.0, 7.0 / 3.0], [7.0 / 6.0, 1.0 / 6.0]], dtype=float),
            vector_field=second_analytic_vector_field,
            eigenfunctions=(
                EigenfunctionSpec(
                    key="phi_lambda_1",
                    title=r"$\phi_{\lambda_1}$",
                    eigenvalue=-1.0,
                    sigmas=(3.0, 3.0),
                    linear_part=(1.0, -2.0),
                    true_eigenfunction=second_analytic_phi_1,
                    nugget=1e-8,
                ),
                EigenfunctionSpec(
                    key="phi_lambda_2",
                    title=r"$\phi_{\lambda_2}$",
                    eigenvalue=2.5,
                    sigmas=(7.0, 7.0),
                    linear_part=(1.0, 1.0),
                    true_eigenfunction=second_analytic_phi_2,
                    nugget=1e-8,
                ),
            ),
        ),
        "duffing": ExperimentSpec(
            key="duffing",
            title="Duffing Oscillator",
            section="Section 5.3",
            description="Kernel reconstruction of the unstable principal eigenfunction for the unforced Duffing oscillator.",
            bounds=((-2.0, 2.0), (-2.0, 2.0)),
            grid_shape=(50, 50),
            linearization=np.array([[0.0, 1.0], [1.0, -0.5]], dtype=float),
            vector_field=duffing_vector_field,
            eigenfunctions=(
                EigenfunctionSpec(
                    key="phi_lambda_unstable",
                    title=r"$\phi_{\lambda_+}$",
                    eigenvalue=(-1.0 + np.sqrt(17.0)) / 4.0,
                    sigmas=(15.0, 15.0),
                    nugget=1e-10,
                ),
            ),
        ),
        "gradient_3d": ExperimentSpec(
            key="gradient_3d",
            title="Three-Dimensional Gradient System",
            section="Section 5.4",
            description="Kernel reconstruction of the unstable principal eigenfunction and a 2D slice of its level set.",
            bounds=((-2.0, 2.0), (-2.0, 2.0), (-2.0, 2.0)),
            grid_shape=(15, 15, 15),
            linearization=np.array(
                [
                    [1.6, -2.2, -0.1],
                    [-2.2, 1.4, -0.1],
                    [-0.1, -0.1, -0.4],
                ],
                dtype=float,
            ),
            vector_field=gradient_system_vector_field,
            eigenfunctions=(
                EigenfunctionSpec(
                    key="phi_lambda_unstable",
                    title=r"$\phi_{\lambda_1}$",
                    eigenvalue=3.7022740721068776,
                    sigmas=(1.1, 1.1, 1.1),
                    nugget=1e-8,
                ),
            ),
            metadata={
                "slice_axis": 2,
                "slice_value": 0.57,
                "equilibrium_guesses": ((0.0, 0.0, 0.0), (1.0, -1.0, 0.0), (-1.0, 1.0, 0.0)),
            },
        ),
    }


def get_experiment(key: str) -> ExperimentSpec:
    registry = build_registry()
    if key not in registry:
        available = ", ".join(sorted(registry))
        raise KeyError(f"Unknown experiment '{key}'. Available experiments: {available}.")
    return registry[key]


def list_experiments() -> tuple[str, ...]:
    return tuple(sorted(build_registry()))


def with_grid_shape(experiment: ExperimentSpec, grid_shape: tuple[int, ...]) -> ExperimentSpec:
    return replace(experiment, grid_shape=grid_shape)

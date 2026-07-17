from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from koopman_hjb.experiments import get_experiment, with_grid_shape
from koopman_hjb.lyapunov import koopman_quadratic_observable, koopman_quadratic_time_derivative


class KoopmanSmokeTests(unittest.TestCase):
    def test_first_analytic_reconstruction_smoke(self) -> None:
        experiment = with_grid_shape(get_experiment("first_analytic"), (12, 12))
        model = experiment.fit("phi_lambda_1")
        learned = experiment.evaluate_on_grid(model)
        truth = experiment.true_on_grid("phi_lambda_1")
        relative_l2_error = np.linalg.norm(learned - truth) / np.linalg.norm(truth)
        self.assertLess(relative_l2_error, 0.35)

    def test_second_analytic_linear_parts_match_the_linearization(self) -> None:
        experiment = get_experiment("second_analytic")
        for spec in experiment.eigenfunctions:
            linear_part = experiment.linear_part_for(spec)
            residual = linear_part @ experiment.linearization - spec.eigenvalue * linear_part
            self.assertLess(np.linalg.norm(residual, ord=np.inf), 1e-12)

    def test_gradient_system_jacobian_matches_finite_difference(self) -> None:
        experiment = get_experiment("gradient_3d")
        origin = np.zeros(experiment.dimension)
        epsilon = 1e-6
        numerical_jacobian = np.zeros_like(experiment.linearization)
        for axis in range(experiment.dimension):
            step = np.zeros(experiment.dimension)
            step[axis] = epsilon
            forward = experiment.vector_field((origin + step)[None, :])[0]
            backward = experiment.vector_field((origin - step)[None, :])[0]
            numerical_jacobian[:, axis] = (forward - backward) / (2.0 * epsilon)
        self.assertTrue(np.allclose(numerical_jacobian, experiment.linearization, atol=1e-6))

    def test_koopman_quadratic_time_derivative_matches_closed_form(self) -> None:
        phi_values = np.array([[1.0, 2.0], [3.0, 4.0]])
        matrix_p = np.array([[2.0, 0.5], [0.5, 1.0]])
        eigenvalues = np.array([-1.0, 3.0])

        observable = koopman_quadratic_observable(phi_values, matrix_p)
        derivative = koopman_quadratic_time_derivative(phi_values, matrix_p, eigenvalues)

        expected_observable = np.array([8.0, 46.0])
        expected_derivative = np.array([24.0, 84.0])
        self.assertTrue(np.allclose(observable, expected_observable))
        self.assertTrue(np.allclose(derivative, expected_derivative))


if __name__ == "__main__":
    unittest.main()

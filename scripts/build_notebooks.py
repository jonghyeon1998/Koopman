from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS_DIR = ROOT / "experiments"


def markdown_cell(text: str):
    return nbf.v4.new_markdown_cell(dedent(text).strip() + "\n")


def code_cell(source: str):
    return nbf.v4.new_code_cell(dedent(source).strip() + "\n")


SETUP_CELL = """
from pathlib import Path
import sys

ROOT = Path.cwd()
if not (ROOT / "src").exists():
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT / "src"))

import matplotlib.pyplot as plt
import numpy as np

np.set_printoptions(precision=6, suppress=True)
plt.rcParams["figure.dpi"] = 140
"""


def notebook_metadata() -> dict:
    return {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python"},
    }


def build_first_analytic() -> nbf.NotebookNode:
    notebook = nbf.v4.new_notebook(metadata=notebook_metadata())
    notebook.cells = [
        markdown_cell(
            """
            # First Analytical Example

            This notebook reproduces the two Section 5.1 eigenfunction experiments from the paper
            using the shared kernel-collocation implementation in `src/koopman_hjb`.
            """
        ),
        code_cell(SETUP_CELL),
        code_cell(
            """
            from koopman_hjb.experiments import get_experiment
            from koopman_hjb.lyapunov import (
                koopman_quadratic_observable,
                koopman_quadratic_time_derivative,
            )
            from koopman_hjb.plotting import plot_comparison_surfaces

            experiment = get_experiment("first_analytic")
            print(experiment.describe())
            x_mesh, y_mesh = experiment.mesh()
            """
        ),
        markdown_cell("## Learned Eigenfunctions"),
        code_cell(
            """
            learned_fields = {}
            eigenvalues = []

            for spec in experiment.eigenfunctions:
                model = experiment.fit(spec.key)
                learned = experiment.evaluate_on_grid(model)
                truth = experiment.true_on_grid(spec.key)
                error = experiment.relative_error_on_grid(spec.key, learned)
                learned_fields[spec.key] = learned
                eigenvalues.append(spec.eigenvalue)

                print(
                    f"{spec.title}: mean relative error = {error.mean():.3e}, "
                    f"max relative error = {error.max():.3e}"
                )
                plot_comparison_surfaces(x_mesh, y_mesh, learned, truth, error, spec.title)
                plt.show()
            """
        ),
        markdown_cell(
            r"""
            ## Quadratic Observable and $\dot{V}(x)$

            This notebook now computes $\dot{V}(x)$ from the learned Koopman eigenfunctions
            using

            $$
            \dot{V}(x)=\sum_{i,j=1}^{d} P_{ij}(\lambda_i+\lambda_j)\phi_i(x)\phi_j(x),
            $$

            rather than explicitly differentiating the learned eigenfunctions.
            """
        ),
        code_cell(
            r"""
            P_matrix = np.eye(len(experiment.eigenfunctions))
            phi_stack = np.stack(
                [learned_fields[spec.key] for spec in experiment.eigenfunctions],
                axis=-1,
            )
            eigenvalues = np.asarray(eigenvalues, dtype=float)

            V_values = koopman_quadratic_observable(phi_stack, P_matrix)
            V_dot_values = koopman_quadratic_time_derivative(
                phi_stack,
                P_matrix,
                eigenvalues,
            )

            figure, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
            panels = (
                (axes[0], V_values, r"Quadratic observable $V(x)$", "viridis"),
                (
                    axes[1],
                    V_dot_values,
                    r"Time derivative $\dot{V}(x)$ from eigenvalues",
                    "coolwarm",
                ),
            )
            for axis, values, title, color_map in panels:
                color = axis.pcolormesh(x_mesh, y_mesh, values, shading="auto", cmap=color_map)
                axis.set_title(title)
                axis.set_xlabel(r"$x_1$")
                axis.set_ylabel(r"$x_2$")
                figure.colorbar(color, ax=axis)
            plt.show()
            """
        ),
    ]
    return notebook


def build_second_analytic() -> nbf.NotebookNode:
    notebook = nbf.v4.new_notebook(metadata=notebook_metadata())
    notebook.cells = [
        markdown_cell(
            """
            # Second Analytical Example

            This notebook reproduces the two Section 5.2 eigenfunction experiments from the paper
            using the shared kernel-collocation implementation in `src/koopman_hjb`.
            """
        ),
        code_cell(SETUP_CELL),
        code_cell(
            """
            from koopman_hjb.experiments import get_experiment
            from koopman_hjb.plotting import plot_comparison_surfaces

            experiment = get_experiment("second_analytic")
            print(experiment.describe())
            x_mesh, y_mesh = experiment.mesh()
            """
        ),
        markdown_cell("## Learned Eigenfunctions"),
        code_cell(
            """
            for spec in experiment.eigenfunctions:
                model = experiment.fit(spec.key)
                learned = experiment.evaluate_on_grid(model)
                truth = experiment.true_on_grid(spec.key)
                error = experiment.relative_error_on_grid(spec.key, learned)

                print(
                    f"{spec.title}: mean relative error = {error.mean():.3e}, "
                    f"max relative error = {error.max():.3e}"
                )
                plot_comparison_surfaces(x_mesh, y_mesh, learned, truth, error, spec.title)
                plt.show()
            """
        ),
    ]
    return notebook


def build_duffing() -> nbf.NotebookNode:
    notebook = nbf.v4.new_notebook(metadata=notebook_metadata())
    notebook.cells = [
        markdown_cell(
            """
            # Duffing Oscillator

            This notebook reproduces the Section 5.3 unstable-eigenfunction experiment for the
            unforced Duffing oscillator.
            """
        ),
        code_cell(SETUP_CELL),
        code_cell(
            """
            from koopman_hjb.experiments import get_experiment
            from koopman_hjb.plotting import plot_duffing_panels

            experiment = get_experiment("duffing")
            spec = experiment.eigenfunctions[0]
            print(experiment.describe())
            print("Target eigenvalue:", spec.eigenvalue)
            x_mesh, y_mesh = experiment.mesh()
            """
        ),
        code_cell(
            """
            model = experiment.fit(spec.key)
            learned = experiment.evaluate_on_grid(model)
            plot_duffing_panels(x_mesh, y_mesh, learned, "Learned unstable eigenfunction")
            plt.show()
            """
        ),
    ]
    return notebook


def build_gradient_3d() -> nbf.NotebookNode:
    notebook = nbf.v4.new_notebook(metadata=notebook_metadata())
    notebook.cells = [
        markdown_cell(
            """
            # Three-Dimensional Gradient System

            This notebook reproduces the Section 5.4 unstable-eigenfunction experiment and plots
            the paper's `z = 0.57` contour slice.
            """
        ),
        code_cell(SETUP_CELL),
        code_cell(
            """
            from koopman_hjb.experiments import get_experiment
            from koopman_hjb.plotting import plot_contour_slice

            experiment = get_experiment("gradient_3d")
            spec = experiment.eigenfunctions[0]
            print(experiment.describe())

            guesses = experiment.metadata["equilibrium_guesses"]
            equilibria = experiment.find_equilibria(guesses)
            print("Computed equilibria:")
            for root in equilibria:
                print(root)
            """
        ),
        code_cell(
            """
            model = experiment.fit(spec.key)
            slice_axis = experiment.metadata["slice_axis"]
            slice_value = experiment.metadata["slice_value"]
            _, _, x_mesh, y_mesh, values = experiment.evaluate_slice(
                model,
                fixed_axis=slice_axis,
                fixed_value=slice_value,
                resolution=200,
            )
            plot_contour_slice(
                x_mesh,
                y_mesh,
                values,
                f"Learned contour slice at x_3 = {slice_value}",
            )
            plt.show()
            """
        ),
    ]
    return notebook


def write_notebook(path: Path, notebook: nbf.NotebookNode) -> None:
    path.write_text(nbf.writes(notebook), encoding="utf-8")


def main() -> None:
    EXPERIMENTS_DIR.mkdir(exist_ok=True)
    write_notebook(
        EXPERIMENTS_DIR / "Koopman_First_Analytic_Example.ipynb",
        build_first_analytic(),
    )
    write_notebook(
        EXPERIMENTS_DIR / "Koopman_Second_Analytic_Example.ipynb",
        build_second_analytic(),
    )
    write_notebook(EXPERIMENTS_DIR / "Koopman_Duffing.ipynb", build_duffing())
    write_notebook(
        EXPERIMENTS_DIR / "Koopman_3D_Gradient_System.ipynb",
        build_gradient_3d(),
    )


if __name__ == "__main__":
    main()

from __future__ import annotations

from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS_DIR = ROOT / "experiments"


def markdown_cell(text: str):
    return nbf.v4.new_markdown_cell(text.strip() + "\n")


def code_cell(source: str):
    return nbf.v4.new_code_cell(source.strip() + "\n")


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
            from koopman_hjb.plotting import plot_comparison_surfaces

            experiment = get_experiment("first_analytic")
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

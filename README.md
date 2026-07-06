# Koopman HJB Experiments

This repository contains the code for the experiments in [Kernels__Koopman__HJB.pdf](Kernels__Koopman__HJB.pdf). The original notebook collection has been cleaned up so that each paper experiment now lives in a single notebook and shares the same reusable kernel-collocation code.

## Repository Layout

- `experiments/`: one notebook per paper experiment.
- `src/koopman_hjb/`: shared solver, experiment definitions, and plotting helpers.
- `scripts/build_notebooks.py`: regenerates the clean notebooks from the shared code layout.
- `tests/test_smoke.py`: small regression checks for the solver and experiment definitions.
- `experiments/Koopman_First_Analytic_Example.ipynb`: Section 5.1.
- `experiments/Koopman_Second_Analytic_Example.ipynb`: Section 5.2.
- `experiments/Koopman_Duffing.ipynb`: Section 5.3.
- `experiments/Koopman_3D_Gradient_System.ipynb`: Section 5.4.

## Setup

Create an environment and install the project dependencies:

```bash
python -m pip install -r requirements.txt
```

The notebooks work from either the repository root or the `experiments/` directory because their setup cell adds `src/` to `sys.path`.

## Running the Experiments

Open any of the four notebooks in `experiments/` and run the cells in order. If you update the shared Python code and want to rebuild the notebook files, run:

```bash
python scripts/build_notebooks.py
```

## Tests

Run the smoke tests with:

```bash
PYTHONPATH=src python -m unittest discover -s tests
```

## Notes on the Refactor

- The duplicated `_1` and `_2` notebooks were merged into one notebook per experiment.
- The shared solver now assembles the kernel collocation system generically for 2D and 3D Gaussian kernels.
- The repository no longer depends on JAX for these experiments; the shared code uses NumPy and SciPy throughout.
- Several notebook inconsistencies were removed while aligning the experiments to the paper, including the split analytic examples, the Duffing notebook drift, and the incorrect 3D potential used in the previous notebook.

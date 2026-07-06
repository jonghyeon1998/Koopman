from __future__ import annotations

import numpy as np
from matplotlib import pyplot as plt


def plot_comparison_surfaces(
    x_mesh: np.ndarray,
    y_mesh: np.ndarray,
    learned: np.ndarray,
    truth: np.ndarray,
    error: np.ndarray,
    title: str,
) -> plt.Figure:
    """Plot learned, true, and error surfaces side by side."""
    figure = plt.figure(figsize=(15, 4.5))
    panels = (
        (learned, f"Learned {title}", "viridis"),
        (truth, f"True {title}", "viridis"),
        (error, "Relative error", "magma"),
    )
    for index, (values, panel_title, color_map) in enumerate(panels, start=1):
        axis = figure.add_subplot(1, 3, index, projection="3d")
        axis.plot_surface(x_mesh, y_mesh, values, cmap=color_map, linewidth=0, antialiased=False)
        axis.set_title(panel_title)
        axis.set_xlabel(r"$x_1$")
        axis.set_ylabel(r"$x_2$")
    figure.tight_layout()
    return figure


def plot_duffing_panels(
    x_mesh: np.ndarray,
    y_mesh: np.ndarray,
    values: np.ndarray,
    title: str,
) -> plt.Figure:
    """Plot the Duffing eigenfunction as a surface and a heat map."""
    figure = plt.figure(figsize=(12, 4.5))

    surface = figure.add_subplot(1, 2, 1, projection="3d")
    surface.plot_surface(x_mesh, y_mesh, values, cmap="viridis", linewidth=0, antialiased=False)
    surface.set_title(title)
    surface.set_xlabel(r"$x_1$")
    surface.set_ylabel(r"$x_2$")

    heatmap = figure.add_subplot(1, 2, 2)
    color = heatmap.pcolormesh(x_mesh, y_mesh, values, cmap="seismic", shading="auto")
    heatmap.set_title(f"{title} (heat map)")
    heatmap.set_xlabel(r"$x_1$")
    heatmap.set_ylabel(r"$x_2$")
    figure.colorbar(color, ax=heatmap)

    figure.tight_layout()
    return figure


def plot_contour_slice(
    x_mesh: np.ndarray,
    y_mesh: np.ndarray,
    values: np.ndarray,
    title: str,
    x_label: str = r"$x_1$",
    y_label: str = r"$x_2$",
) -> plt.Figure:
    """Plot a filled contour slice of a 3D eigenfunction."""
    figure, axis = plt.subplots(figsize=(6, 5))
    contour = axis.contourf(x_mesh, y_mesh, values, levels=20, cmap="viridis")
    axis.set_title(title)
    axis.set_xlabel(x_label)
    axis.set_ylabel(y_label)
    figure.colorbar(contour, ax=axis)
    figure.tight_layout()
    return figure

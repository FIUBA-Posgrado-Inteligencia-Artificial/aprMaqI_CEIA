import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap


def plot_boundary(  # noqa: PLR0913
    x_data,
    y_data,
    model,
    step_x=(0.1, 0.1),
    max_x=(1, 1),
    min_x=(-1, -1),
    point_size=5,
    figsize=(7, 5),
    label_point=("1", "0"),
    colormap_frontier=("#7aa5fb", "#f8b389"),
    colormap_points=("#5471ab", "#d1885c"),
    labels_axis=("x1", "x2"),
    legend=True,
    legend_title=None,
    marker="o",
):
    # Hotfix for bug in matplotlib 3.8.0.
    # https://github.com/matplotlib/matplotlib/issues/26949/
    if type(colormap_frontier) is tuple:
        colormap_frontier = list(colormap_frontier)
    if type(colormap_points) is tuple:
        colormap_points = list(colormap_points)

    # Crear la malla de puntos para el gráfico
    x1, x2 = np.meshgrid(
        np.arange(
            start=x_data[:, 0].min() + min_x[0],
            stop=x_data[:, 0].max() + max_x[0],
            step=step_x[0],
        ),
        np.arange(
            start=x_data[:, 1].min() + min_x[1],
            stop=x_data[:, 1].max() + max_x[1],
            step=step_x[1],
        ),
    )
    x_cont = np.array([x1.ravel(), x2.ravel()]).T

    # Crear el gráfico de contorno
    plt.figure(figsize=figsize)
    plt.contourf(
        x1,
        x2,
        model.predict(x_cont).reshape(x1.shape),
        alpha=0.75,
        cmap=ListedColormap(colormap_frontier),
    )
    plt.xlim(x1.min(), x1.max())
    plt.ylim(x2.min(), x2.max())

    # Graficar los puntos de entrenamiento
    for i, j in enumerate(np.unique(y_data)):
        plt.scatter(
            x_data[y_data == j, 0],
            x_data[y_data == j, 1],
            c=colormap_points[i],
            label=label_point[i],
            s=point_size,
            marker=marker,
        )

    plt.xlabel(labels_axis[0])
    plt.ylabel(labels_axis[1])
    if legend:
        plt.legend(title=legend_title)


def plot_svm_margins(  # noqa: PLR0913
    x_data,
    model,
    max_x=(1, 1),
    min_x=(-1, -1),
    step_x=(0.1, 0.1),
    colors="k",
    alpha=1,
    linestyles=("--", "-", "--"),
    linewidths=(1, 3, 1),
):
    # Crear la malla de puntos para el gráfico
    x1, x2 = np.meshgrid(
        np.arange(
            start=x_data[:, 0].min() + min_x[0],
            stop=x_data[:, 0].max() + max_x[0],
            step=step_x[0],
        ),
        np.arange(
            start=x_data[:, 1].min() + min_x[1],
            stop=x_data[:, 1].max() + max_x[1],
            step=step_x[1],
        ),
    )
    x_cont = np.array([x1.ravel(), x2.ravel()]).T

    z = model.decision_function(x_cont)

    if len(z.shape) == 1:
        z = z.reshape(-1, 1)

    for col in range(z.shape[1]):
        plt.contour(
            x1,
            x2,
            z[:, col].reshape(x1.shape),
            colors=colors,
            levels=[-1, 0, 1],
            alpha=alpha,
            linestyles=linestyles,
            linewidths=linewidths,
        )

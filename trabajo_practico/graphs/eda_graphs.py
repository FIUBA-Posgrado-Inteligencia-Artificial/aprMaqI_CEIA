import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd


def plot_service_distribution(df, service_columns):
    category_order = ["No aplica", 1, 2, 3, 4, 5]

    fig, axes = plt.subplots(5, 3, figsize=(15, 24), sharey=True)
    axes = axes.flatten()

    for i, col in enumerate(service_columns):
        ax = axes[i]

        order = [value for value in category_order if value in df[col].unique()]

        sns.countplot(
            data=df,
            x=col,
            hue=col,
            order=order,
            hue_order=order,
            palette="pastel",
            legend=False,
            ax=ax,
        )

        ax.set_xlabel(col)
        ax.set_ylabel("Cantidad")
        ax.tick_params(axis="x", labelrotation=20)

    for ax in axes[len(service_columns) :]:
        fig.delaxes(ax)

    plt.suptitle(
        "Distribución de las valoraciones de los servicios", fontsize=14, y=1.01
    )

    plt.tight_layout()
    plt.show()


def plot_kendall(df):
    corr_kendall = df.corr(method="kendall")

    plt.figure(figsize=(14, 11))

    mask = np.triu(np.ones_like(corr_kendall, dtype=bool))

    sns.heatmap(
        corr_kendall,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        vmin=-1,
        vmax=1,
    )

    plt.title(
        "Kendall Tau-b entre valoraciones de servicios\n" "(excluyendo No aplica)"
    )
    plt.tight_layout()
    plt.show()


def plot_crosstab(left: pd.Series, right: pd.Series):
    plt.figure(figsize=(8, 5))
    cross_tab = pd.crosstab(left, right)
    sns.heatmap(cross_tab, annot=True, fmt="d", cmap="Blues")
    plt.title(f"Tabla de contingencia ({left.name} - {right.name})")
    plt.xlabel(right.name)
    plt.ylabel(left.name)
    plt.show()

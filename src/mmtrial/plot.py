"""Plot training loss and metrics from a TSV file."""

import argparse

import matplotlib.pyplot as plt
import pandas as pd


def _plot(  # noqa: PLR0913
    df: pd.DataFrame,
    title: str,
    ylabel: str,
    metric1: str,
    metric2: str,
    label1: str,
    label2: str,
    n: int,
) -> None:
    """Read the training log dataframe and plot the Loss and metrics.

    Args:
        df (pd.DataFrame): The training log dataframe.
        title (str): The title of the plot.
        ylabel (str): The label for the y-axis.
        metric1 (str): The first metric to plot.
        metric2 (str): The second metric to plot.
        label1 (str): The label for the first metric.
        label2 (str): The label for the second metric.
        n (int): The number of epochs to plot.
    """
    df = df[:n]
    # Combined plot with Loss and mAP
    fig, ax1 = plt.subplots(figsize=(12, 8))
    ax2 = ax1.twinx()

    color = "tab:red"
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss", color=color)
    ax1.plot(df["Epoch"], df["Loss"], color=color, label="Loss")
    ax1.tick_params(axis="y", labelcolor=color)
    ax1.grid(visible=True)

    color = "tab:blue"
    ax2.set_ylabel(ylabel, color=color)
    ax2.plot(df["Epoch"], df[metric1], label=label1, color="blue")
    ax2.plot(df["Epoch"], df[metric2], label=label2, color="green")
    ax2.tick_params(axis="y", labelcolor=color)

    # Adjust layout to prevent title from being cut off
    plt.title(title)
    # Adjust the top margin to make room for the title
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Combine legends from both axes
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    fig.legend(
        lines_1 + lines_2,
        labels_1 + labels_2,
        loc="upper left",
        bbox_to_anchor=(0.1, 0.9),
    )

    # Save the plot as a file with higher resolution
    plt.savefig("loss_metric.png", dpi=300)


def plot_loss_map(tsv: str, title: str, n: int) -> None:
    """Read the training TSV and plot the Loss and mAP metrics."""
    # ruff: noqa: PD901
    df = pd.read_csv(tsv, sep="\t")
    df["Epoch"] = range(1, len(df) + 1)

    # Convert mAP values to percentages
    metric1, metric2 = "mAP50", "mAP75"
    df[metric1] = df[metric1] * 100
    df[metric2] = df[metric2] * 100

    return _plot(df, title, "mAP (%)", "mAP50", "mAP75", "mAP@50", "mAP@75", n)


def plot_loss_acc(tsv: str, title: str, n: int) -> None:
    """Read the training TSV and plot the Loss and accuracy metrics."""
    # ruff: noqa: PD901
    df = pd.read_csv(tsv, sep="\t")
    df["Epoch"] = range(1, len(df) + 1)

    return _plot(df, title, "Acc (%)", "acctop1", "acctop2", "Top1", "Top2", n)


if __name__ == "__main__":
    # Set up argument parsing to accept the file path as a cmd argument
    parser = argparse.ArgumentParser(
        description="Plot training loss and metrics from a TSV file")
    parser.add_argument(
        "tsv",
        type=str,
        help="Path to the TSV file containing the training data",
    )
    parser.add_argument(
        "title",
        type=str,
        help="Title of the plot",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="map",
        choices=["map", "acc"],
        help="Metric to plot (default: map)",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=100,
        help="Number of epochs to plot (default: 100)",
    )
    args = parser.parse_args()

    func = plot_loss_map if args.metric == "map" else plot_loss_acc

    # Call the function with the provided file path
    func(args.tsv, args.title, args.n)

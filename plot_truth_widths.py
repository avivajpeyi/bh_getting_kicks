# -*- coding: utf-8 -*-
"""Module one liner

This module does what....

Example usage:

"""
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import rcParams

from bbh_simulator.corner_kwargs import BILBY_BLUE_COLOR, VIOLET_COLOR

rcParams["font.size"] = 20
rcParams["font.family"] = "serif"
rcParams["font.sans-serif"] = ["Computer Modern Sans"]
rcParams["text.usetex"] = True
rcParams['axes.labelsize'] = 30
rcParams['axes.titlesize'] = 30
rcParams['axes.labelpad'] = 20


def plot():
    """Plots histogram"""
    df = pd.read_csv("output/quantiles_widths.csv")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].hist(df.original, color=BILBY_BLUE_COLOR, alpha=0.5, label="Original",
                 density=True)
    axes[0].hist(df.reweighted, color=VIOLET_COLOR, alpha=0.5, label="Reweighted",
                 density=True)
    axes[0].legend(fontsize=18, frameon=False)
    axes[0].set_xlabel(r"$\chi_p\  68\%$ C.I. Width")
    axes[0].set_ylabel("Probability Density")
    axes[0].grid()
    fraction = df.reweighted / df.original
    axes[1].hist(fraction, color='teal', alpha=0.2, label="", density=True)
    axes[1].set_xlabel("New Width / Original Width")
    axes[1].set_ylabel("Probability Density")
    axes[1].grid()

    width = df.reweighted - df.original
    axes[2].scatter(df.index, df.reweighted - df.original, color="teal", alpha=0.5, label="Original")
    axes[2].set_xlabel("Injection Id")
    axes[2].set_ylabel(r"$\Delta$ Width")
    axes[2].grid()

    plt.tight_layout()
    plt.savefig("chi_p_posterior_width.png")


def main():
    plot()


if __name__ == "__main__":
    main()

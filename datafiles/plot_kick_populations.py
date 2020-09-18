# -*- coding: utf-8 -*-
"""Module one liner

This module does what....

Example usage:

"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats


AGN = "agn_population.csv"
ALIGNED = "aligned_spin_population.csv"
PRECESSING = "precessing_spin_population.csv"

POPULATIONS = {
    "AGN BBHs": [AGN, 75],
    "Precessing-Spin BBHs": [PRECESSING, 75],
    "Aligned-Spin BBHs": [ALIGNED, 15],

}


def plot():
    fig, ax = plt.subplots(2, sharex=True)
    for label, csv in POPULATIONS.items():
        data = pd.read_csv(csv[0])
        plot_data = list(data.remnant_kick.values)
        ax[0].hist(plot_data, density=True, label=label, bins=csv[1], alpha=0.5)
        ax[1].hist(plot_data, density=True, label=label, bins=csv[1], cumulative=-1, alpha=0.5, histtype ='step')

    ax[0].legend()
    ax[0].set_ylabel("Density")
    ax[1].set_ylabel("1-CDF")
    ax[1].set_xlabel("Remnant Kick")
    ax[1].set_xlim(left=0)
    fig.savefig("kick_distributions.png")

    fig, ax = plt.subplots()
    agn_cos_tilt = pd.read_csv(AGN).cos_tilt_2
    precessing_cos_tilt = pd.read_csv(PRECESSING).cos_tilt_2
    ax.hist(agn_cos_tilt, label="AGN BBHs", density=True, alpha=0.5)
    ax.hist(precessing_cos_tilt, label="Precessing BBHs", density=True, alpha=0.5)
    ax.set_ylabel("Density")
    ax.set_xlabel("BH cos(tilt)")
    ax.legend()
    fig.savefig("bbh_tilt.png")


def main():
    plot()


if __name__ == "__main__":
    main()

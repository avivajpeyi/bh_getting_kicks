# -*- coding: utf-8 -*-
"""plot kicks from differnt samples
"""

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import rcParams

rcParams["font.size"] = 20
rcParams["font.family"] = "serif"
rcParams["font.sans-serif"] = ["Computer Modern Sans"]
rcParams["text.usetex"] = True
rcParams['axes.labelsize'] = 30
rcParams['axes.titlesize'] = 30
rcParams['axes.labelpad'] = 20

SAMPLES = "datafiles/{}_population.csv"
DATA = dict(agn=dict(bins=100, label="AGN"),
            aligned_spin=dict(bins=12, label="Aligned Spin"),
            precessing_spin=dict(bins=100, label="Precessing Spin"),
            )
LABELS = list(DATA.keys())


def plot():
    """Plots a corner plot from some samples."""
    fig, ax = plt.subplots()
    for key, data in DATA.items():
        samples = pd.read_csv(SAMPLES.format(key))
        ax.hist(
            samples['remnant_kick'],
            bins=data['bins'], density=True, alpha=0.5,
            label=data['label']
        )
    ax.legend(frameon=False)
    ax.set_ylabel("Density")
    ax.set_xlabel("Kick Magnitude (km/s)")
    fig.savefig("kick_magnitudes_for_different_populations.png")


def main():
    plot()


if __name__ == "__main__":
    main()

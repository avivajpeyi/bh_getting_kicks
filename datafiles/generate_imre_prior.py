# -*- coding: utf-8 -*-
"""Module one liner

This module does what....

Example usage:

"""
import math

import bilby
import corner
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import scipy.interpolate
from matplotlib import rcParams

from corner_kwargs import CORNER_KWARGS

NUM_INTERPOLATED_POINTS = 10000

rcParams["font.size"] = 16
rcParams["font.family"] = "serif"
rcParams["font.sans-serif"] = ["Computer Modern Sans"]
rcParams["text.usetex"] = True
rcParams['axes.labelsize'] = 16
rcParams['axes.titlesize'] = 30
rcParams['axes.labelpad'] = 25

LABEL_LATEX = dict(
    a=r'$|a|$',
    cos_theta_a=r'$\cos \theta_a$',
    cos_theta_jn=r'$\cos \theta_{jn}$',
    cos_tilt=r'$\cos tilt$'
)


def get_cubic_fit(data: pd.DataFrame):
    data = data.sort_values(by='x')
    data_x, data_y = data['x'].dropna(), data['y'].dropna()
    f = scipy.interpolate.CubicSpline(data_x, data_y)
    x = list(np.linspace(min(data_x), max(data_x), NUM_INTERPOLATED_POINTS))
    data = pd.DataFrame(dict(x=x, y=f(x)))
    return data.sample(frac=0.1, weights=data.y)


def get_interpolated_data(data, label):
    orig_x, orig_prob = data[label], data[f"{label}_prob"]
    data = pd.DataFrame(dict(x=orig_x, y=orig_prob))
    data = get_cubic_fit(data)
    return pd.DataFrame({label: data['x'], f"{label}_prob": data['y']})


def plot_data(data, ax, label):
    xx = data[label].values
    min_x, max_x = min(xx), max(xx)
    ax.set_xlim(math.floor(min_x), math.ceil(max_x))
    ax.set_aspect('auto')
    n, bins = np.histogram(xx, density=True, bins=20)
    width = np.diff(bins)
    area = sum(width * n)
    center = (bins[:-1] + bins[1:]) / 2
    ax.bar(center, n, align='center', width=width, alpha=0.1)
    ax.plot(center, n, label=f"Area {area:.2f}")
    ax.legend(fontsize=12)
    ax.set_xlabel(LABEL_LATEX[label])
    return center, n


def write_data(x, y, fname):
    data = np.array([x, y]).T
    np.savetxt(X=data, fname=fname)


def load_imre_data():
    raw_data = pd.read_csv("imre_prior_data.csv")
    data = pd.DataFrame()
    data['cos_theta_a'] = get_interpolated_data(raw_data, label='cos_theta_a')[
        'cos_theta_a'].values
    data['a'] = get_interpolated_data(raw_data, label='a')['a'].values
    data['cos_theta_jn'] = np.random.uniform(low=-1.0, high=1.0, size=len(data))
    data['theta_a'] = np.arccos(data['cos_theta_a'])
    data['theta_jn'] = np.arccos(data['cos_theta_jn'])
    data['tilt'] = data['theta_a'] - data['theta_jn']
    data['cos_tilt'] = np.cos(data['tilt'])
    return data


def plot():
    """Plots a scatter plot."""
    data = load_imre_data()
    num_plots = 4
    fig, ax = plt.subplots(1, num_plots, sharey=True, figsize=(3 * num_plots, 3))
    ax[0].set_ylabel('PDF')
    for i, l in enumerate(["a", "cos_theta_a", "cos_theta_jn", "cos_tilt"]):
        x, prob = plot_data(data, ax[i], l)
        write_data(x, prob, fname=f"{l}_prior.txt")
    plt.tight_layout()
    fig.savefig('agn_data/test.png')


def load_bilby_prior():
    prior = bilby.prior.PriorDict(dictionary=
    {
        'a': bilby.core.prior.FromFile(file_name="a_prior.txt", minimum=0,
                                       maximum=1,
                                       name='a', latex_label=r'$|a|$'),
        'cos tilt': bilby.core.prior.FromFile(file_name="cos_theta_a_prior.txt",
                                              minimum=-1.0, maximum=+1.0,
                                              name='cos tilt',
                                              latex_label=r'$\cos tilt$')
    }
    )
    samples = pd.DataFrame(prior.sample(1000))
    corner.corner(samples, **CORNER_KWARGS,
                  labels=[p.latex_label for p in prior.values()])
    plt.savefig("agn_data/test_2.png")
    samples.to_csv("agn_population.csv")


def main():
    plot()
    load_bilby_prior()


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""Reweight results based on kick distribution

We know kick v ~ 200km/s +/- 50km/s

We can re-weight our samples based on this

Calculate the prob(kick) based on the above distribution

re-weight our samples by np.random.choice(samples, fraction=0.1, weights=prob(kick_for_sample))

Example usage:

"""
import argparse
import logging
import os
import sys

import corner
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from bilby.gw import conversion
from scipy.stats import norm

logging.getLogger().setLevel(logging.INFO)

KICK_WEIGHT = "kick_weight"

CORNER_KWARGS = dict(smooth=0.9, label_kwargs=dict(fontsize=16),
                     title_kwargs=dict(fontsize=16), color='#0072C1',
                     truth_color='tab:orange', quantiles=[0.16, 0.84],
                     levels=(1 - np.exp(-0.5), 1 - np.exp(-2), 1 - np.exp(-9 / 2.)),
                     plot_density=False, plot_datapoints=False, fill_contours=True,
                     show_titles=True,
                     max_n_ticks=3, )


def parse_cli_args(args=sys.argv[1:]):
    parser = argparse.ArgumentParser(description='Reweight samples based on true kick')
    parser.add_argument('--result', type='str', required=True,
                        help='A CSV file of the samples')
    parser.add_argument('--outfile', type='str', required=True,
                        help='Reweighted output samples CSV filename')
    parser.add_argument('--kick-mean', type='float', required=True,
                        help='The mean value of the true kick distribution (im km/s)')
    parser.add_argument('--kick-stddev', type='float', required=True,
                        help='The mean value of the true kick distribution (im km/s)')

    parsed_args = parser.parse_args(args)

    if not os.path.isfile(parsed_args.result):
        raise FileNotFoundError(
            f"Result {parsed_args.result} is not a csv file that can be accessed"
        )

    if os.path.isfile(parsed_args.outfile):
        logging.warning(f"{parsed_args.outfile} already exists and will be overwritten."
                        f"Terminate job if you do not want to overwrite.")

    return parsed_args


def load_samples(result_file):
    """Loads samples from result file"""
    samples = pd.read_csv(result_file)
    samples = conversion.generate_component_spins(samples)
    try:
        assert len(samples.remnant_kick_mag) > 0
    except Exception as e:
        raise ValueError(
            f"Error: {e}. "
            f"There are no `remnant_kick_mag` samples. "
            f"The samples present are for {list(samples.columns.values)}")
    logging.info(f"Loaded {result_file}")
    return samples


def add_new_kick_distribution_weights(samples, kick_mean, kick_sigma):
    logging.info(f"Adding kick weights based on N({kick_mean}, {kick_sigma})")
    kick_prior = norm(loc=kick_mean, scale=kick_sigma)
    samples[KICK_WEIGHT] = np.abs(np.exp(kick_prior.logpdf(samples.remnant_kick_mag)))
    return samples


def plot_corner(samples, f, weights=False):
    logging.info(f"Plotting {f}")
    s = samples[
        ["mass_1_source_new", "mass_2_source_new", "remnant_new_source_m_remnant_mass",
         "chi_p", "chi_eff", "mass_ratio", "tilt_1", "tilt_2",
         "luminosity_distance", "redshift", "remnant_new_source_m_remnant_kick_mag"]]
    corner_kwargs = CORNER_KWARGS.copy()
    corner_kwargs.update(dict(
        labels=[r"$m_1$(source)", r"$m_2$(source)", r"$m_rem$(source)", r"$\chi_p$",
                r"$\chi_{eff}$", "q", "tilt_1", "tilt_2", r"$d_l$",
                r"$z$",
                r"$|\vec{v}_k|$ km/s"]))

    if weights:
        corner.corner(s, weights=samples.kick_weight, **corner_kwargs)
    else:
        corner.corner(s, **corner_kwargs)
    plt.suptitle(f"{f}", fontsize=30, fontdict=dict(color='darkblue'))
    plt.savefig(f"{f}.png")
    plt.close()


def main():
    samples = load_samples(result_file="NRsur(PHM)_pesummary_with_kicks.dat")
    plot_corner(samples, f="No Reweighting")

    samples = add_new_kick_distribution_weights(samples, kick_mean=200, kick_sigma=50)
    plot_corner(samples, f="kick N(200, 50)", weights=True)

    samples = add_new_kick_distribution_weights(samples, kick_mean=200, kick_sigma=100)
    plot_corner(samples, f="kick N(200, 100)", weights=True)

    samples = add_new_kick_distribution_weights(samples, kick_mean=200, kick_sigma=200)
    plot_corner(samples, f="kick N(200, 200)", weights=True)


if __name__ == "__main__":
    main()

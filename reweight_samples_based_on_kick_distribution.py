# -*- coding: utf-8 -*-
"""Reweight results based on kick distribution

We know kick v ~ 200km/s +/- 50km/s

We can re-weight our posterior based on this

Calculate the prob(kick) based on the above distribution

re-weight our posterior by np.random.choice(posterior, fraction=0.1, weights=prob(kick_for_sample))

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

from corner_kwargs import CORNER_KWARGS

logging.getLogger().setLevel(logging.INFO)

KICK_WEIGHT = "kick_weight"

PARAMS = dict(
    mass_1_source=r'$m_1$(source)',
    mass_2_source=r'$m_2$(source)',
    remnant_mass=r'$m_rem$(source)',
    chi_p=r'$\\chi_p$',
    chi_eff=r'$\\chi_{eff}$',
    mass_ratio=r'q',
    tilt_1=r'tilt_1',
    tilt_2=r'tilt_2',
    luminosity_distance=r'$d_l$',
    redshift=r'$z$',
    remnant_kick_mag=r'$|\\vec{v}_k|$ km/s'
)


def parse_cli_args(args=sys.argv[1:]):
    parser = argparse.ArgumentParser(
        description='Reweight posterior based on true kick')
    parser.add_argument('--samples-csv', type=str, required=True,
                        help='A CSV file of the posterior')
    parser.add_argument('--kick-mean', type=float, required=False,
                        help='The mean value of the true kick distribution (im km/s)')
    parser.add_argument('--kick-stddev', type=float, required=False,
                        help='The mean value of the true kick distribution (im km/s)')
    parser.add_argument('--true-file', type=str, required=False,
                        help='CSV File with the True Vals of the parameters')
    parser.add_argument('--true-idx', type=int, required=False,
                        help='Row idx of the true val')

    parsed_args = parser.parse_args(args)

    return parsed_args


class Samples:
    def __init__(self, samples_csv, kick_mean, kick_sigma, truths):
        self.truths = truths
        self.posterior = self.load_posterior(samples_csv)
        self.add_new_kick_distribution_weights(kick_mean, kick_sigma)

    @staticmethod
    def load_posterior(result_file):
        """Loads posterior from result file"""
        samples = pd.read_csv(result_file)
        samples = conversion.generate_component_spins(samples)
        try:
            assert len(samples.remnant_kick_mag) > 0
        except Exception as e:
            raise ValueError(
                f"Error: {e}. "
                f"There are no `remnant_kick_mag` posterior. "
                f"The posterior present are for {list(samples.columns.values)}")
        logging.info(f"Loaded {result_file}")
        return samples

    def add_new_kick_distribution_weights(self, kick_mean, kick_sigma):
        logging.info(f"Adding kick weights based on N({kick_mean}, {kick_sigma})")
        kick_prior = norm(loc=kick_mean, scale=kick_sigma)
        self.posterior[KICK_WEIGHT] = np.abs(
            np.exp(kick_prior.logpdf(self.posterior.remnant_kick_mag)))

    def plot_corner(self, f, weights=False):
        logging.info(f"Plotting {f}")
        s = self.posterior[list(PARAMS.keys())]
        corner_kwargs = CORNER_KWARGS.copy()
        corner_kwargs.update(dict(labels=list(PARAMS.values())))
        if len(self.truths) > 0:
            corner_kwargs.update(dict(truths=[self.truths[k] for k in PARAMS.keys()]))
        if weights:
            corner.corner(s, weights=self.posterior.kick_weight, **corner_kwargs)
        else:
            corner.corner(s, **corner_kwargs)
        plt.suptitle(f"{f}", fontsize=30, fontdict=dict(color='darkblue'))
        plt.savefig(f"{f}.png")
        plt.close()


def get_truth_values(truth_csv, truth_idx):
    return pd.read_csv(truth_csv).iloc[truth_idx].to_dict()


def validate_cli_args(parsed_args):
    assert os.path.isfile(
        parsed_args.samples_csv), \
        f"Result {parsed_args.samples_csv} is not a csv file that can be accessed"

    if parsed_args.true_file is not None:
        assert parsed_args.true_idx is not None, \
            "A idx for the true values must be passed"
        assert os.path.isfile(
            parsed_args.true_file), f"True file {parsed_args.true_file} cant be accessed"


def main():
    args = parse_cli_args()
    validate_cli_args(args)
    if args.true_file:
        truths = get_truth_values(args.true_file, args.true_idx)
        args.kick_mean = truths["remnant_kick_mag"]
        args.kick_sigma = 50
    else:
        truths = {}
    samples = Samples(samples_csv=args.samples_csv, kick_mean=args.kick_mean,
                      kick_sigma=args.kick_sigma, truths=truths)

    fname = args.result.replace(".csv", "no_reweighting_corner.png")
    samples.plot_corner(f=fname)

    fname = args.result.replace(".csv", "kick_mu{}_sigma{}_corner.png")
    samples.plot_corner(f=fname, weights=True)


if __name__ == "__main__":
    main()

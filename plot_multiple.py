# -*- coding: utf-8 -*-
"""Module one liner

This module does what....

Example usage:

"""
import numpy as np
import matplotlib.pyplot as plt
import plot_corner_weighted_with_kick
import tqdm
import argparse
import pandas as pd

FILE = "/fred/oz117/avajpeyi/projects/phase-marginalisation-test/jobs/out_hundred_injections_gstar/out_injection_{num}/result/injection_{num}_0_posterior_samples_with_kicks.dat"


def main():
    truths = pd.read_csv('datafiles/injections.csv')
    for index, truth in tqdm.tqdm(truths.iterrows(), total=len(truths)):
        f = FILE.format(num=index)
        kick_mean = truth['remnant_kick_mag']
        kick_sigma = 50

        samples = plot_corner_weighted_with_kick.Samples(
            samples_csv=f,
            kick_mean=kick_mean,
            kick_sigma=kick_sigma,
            truths=truth.to_dict()
        )
        fname1 = f.replace(".dat", "_no_reweighting_corner.png")
        samples.plot_corner(f=fname1, title="No Reweigting")

        fname2 = f.replace(
            ".dat",
            f"_kick_mu{int(kick_mean)}_sigma{int(kick_sigma)}_corner.png"
        )
        samples.plot_corner(
            f=fname2, weights=True,
            title=r"Reweighted with {}".format(
                plot_corner_weighted_with_kick.get_normal_string(
                    int(kick_mean),
                    int(kick_sigma)
                )
            ))

        fname3 = f.replace(".dat", "_overlaid.png")
        samples.plot_overlaid_corner(fname3)

        fname4 = f.replace(".dat", "_corner.png")
        plot_corner_weighted_with_kick.combine_images_horizontally(
            [fname3, fname1, fname2], f=fname4)


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""Module one liner

This module does what....

Example usage:

"""
import pandas as pd
import tqdm

import plot_corner_weighted_with_kick

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
        try:
            plot_corner_weighted_with_kick.process_sample(samples, kick_mean,
                                                          kick_sigma, f)
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    main()

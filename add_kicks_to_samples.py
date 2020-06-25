"""Python module to load samples and calculate their kicks"""

import os

import pandas as pd
import tqdm
from .black_hole import merge_bbh_pair, BlackHole


class Samples:
    def __init__(self, filename: str):
        self.filename = filename
        self.posterior = self.read_dat_file(filename)

    @staticmethod
    def read_dat_file(filename: str):
        assert os.path.isfile(filename)
        posterior = pd.read_csv(filename, "\t")
        assert len(posterior.columns.values) > 2
        posterior["id"] = posterior.index + 1
        posterior = posterior.set_index('id')
        return posterior

    def calculate_remnant_kick_velocity(self):
        p = self.posterior
        remnant_data = []
        progress_bar = tqdm.tqdm(p.iterrows(), total=len(p), desc="Calculating Kicks")
        for idx, sample in progress_bar:
            remnant_dict = get_sample_kick(sample)
            remnant_dict = {f"remnant_{k}": v for k, v in remnant_dict.items()}
            remnant_dict.update(dict(id=idx))
            remnant_data.append(remnant_dict)
        remnant_df = pd.DataFrame(remnant_data)
        remnant_df = remnant_df.set_index("id")
        self.posterior = p.merge(remnant_df, on="id")

    def save_samples_with_kicks(self):
        self.calculate_remnant_kick_velocity()
        filename = self.filename.replace(".dat", "_with_kicks.dat")
        self.posterior.to_csv(filename)


def get_sample_kick(s):
    """

    :param s: A posterior sample
    :return:
    """
    remnant = merge_bbh_pair(
        bh_1=BlackHole(mass=s.mass_1, spin=[s.spin_1x, s.spin_1y, s.spin_1z]),
        bh_2=BlackHole(mass=s.mass_2, spin=[s.spin_2x, s.spin_2y, s.spin_2z]),
    )
    return remnant.to_dict()


if __name__ == "__main__":
    samples = Samples(filename="samples.dat")
    samples.save_samples_with_kicks()

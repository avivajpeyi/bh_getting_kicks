from typing import List, Optional

from .utils import mag, c
import numpy as np

PM = u"\u00B1"  # ±
CHI = u"\u03C7"  # χ

class BlackHole:
    bh_counter = 0

    def __init__(self,
                 mass: float,
                 spin: List[float],
                 parents: Optional = None,
                 kick: Optional[float] = [0,0,0],
                 mass_unc: Optional[float] = None,
                 spin_unc: Optional[List[float]] = None,
                 kick_unc: Optional[List[float]] = [0,0,0]
                 ):
        self.mass = mass
        self.spin = spin
        self.kick = kick
        self.mass_unc = mass_unc
        self.spin_unc = spin_unc
        self.kick_unc = kick_unc
        self.parents = parents
        self.generation_number = self._get_generation_number()
        self.id = BlackHole.bh_counter
        BlackHole.bh_counter += 1

    def __str__(self):
        if self.mass_unc:
            vk = mag(self.kick) * c
            vk_unc = mag(self.kick_unc) * c

            return (
                f"Gen# {self.generation_number}\n"
                f"Mass {self.mass:.2f} {PM} {self.mass_unc:.2f}\n"
                f"|{CHI}| {mag(self.spin):.2f} {PM} {mag(self.spin_unc):.2f}\n"
                f"|vk| {vk:.2f}  {PM} {vk_unc:.2f} km/s"
            )
        else:
            return (
                f"Gen# {self.generation_number}\n"
                f"Mass {self.mass:.2f}\n"
                f"|{CHI}| {mag(self.spin):.2f}"
            )

    def __repr__(self):
        return self.__str__()

    def to_dict(self):
        return dict(
            mass=self.mass,
            spin_mag=mag(self.spin),
            spin_x=self.spin[0], spin_y=self.spin[1], spin_z=self.spin[2],
            kick_mag=mag(self.kick)*c,
            kick_x=self.kick[0]*c, kick_y=self.kick[1]*c, kick_z=self.kick[2]*c,
            spin_unc_x=self.spin_unc[0], kick_unc_x=self.kick_unc[0]*c,
            spin_unc_y=self.spin_unc[1], kick_unc_y=self.kick_unc[1]*c,
            spin_unc_z=self.spin_unc[2], kick_unc_z=self.kick_unc[2]*c,
        )

    @property
    def parents(self):
        return self._parents

    @parents.setter
    def parents(self, parents):
        if parents:
            assert len(parents) == 2
        self._parents = parents

    @property
    def spin(self):
        return self._spin

    @spin.setter
    def spin(self, spin):
        assert len(spin) == 3
        self._spin = spin

    def _get_generation_number(self):
        if not self.parents:
            return 0
        else:
            return self.parents[0].generation_number + 1

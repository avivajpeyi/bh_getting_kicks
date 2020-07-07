import logging
import os
from typing import List, Tuple, Dict

import bilby
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import surfinBH
import tqdm

from . import utils
from .black_hole import BlackHole


logging.getLogger().setLevel(logging.INFO)

POPULATION_PRIOR = os.path.join(os.path.dirname(__file__), "init_population.prior")


class BlackHolePopulation:
    def __init__(self, number_of_initial_bh):
        self.number_of_initial_bh = number_of_initial_bh
        self.population = {}
        self.__init_population()

    @property
    def population_size(self):
        return len(self.population)

    @property
    def number_of_generation(self):
        return len(self.get_generation_counts())

    @property
    def number_of_initial_bh(self):
        return self._number_of_initial_bh

    @number_of_initial_bh.setter
    def number_of_initial_bh(self, number_of_initial_bh):
        if not utils.is_power_of_two(number_of_initial_bh):
            raise ValueError(f"{number_of_initial_bh} needs to be a power of 2.")
        self._number_of_initial_bh = number_of_initial_bh

    def __init_population(self):
        logging.info(f"Initialising {self.number_of_initial_bh} BH")
        BlackHole.bh_counter = 0
        bbh_pop_prior = BBHPriorDict(filename=POPULATION_PRIOR)(filename=POPULATION_PRIOR)
        num_bbh_pairs = self.number_of_initial_bh // 2
        bbh_population_data = bbh_pop_prior.sample(num_bbh_pairs)
        self.population = {}
        for i in range(num_bbh_pairs):
            bh_1 = BlackHole(
                mass=bbh_population_data["mass_1"][i],
                spin=[0, 0, bbh_population_data["a_1"][i]],
            )
            bh_2 = BlackHole(
                mass=bbh_population_data["mass_2"][i],
                spin=[0, 0, bbh_population_data["a_2"][i]],
            )
            self.population.update({bh.id: bh for bh in [bh_1, bh_2]})

    def conduct_multiple_mergers(self):
        merging_bh_list = list(self.population.values())
        while len(merging_bh_list) > 1:
            merging_bh_list = self.__truncate_bh_list(merging_bh_list)
            remnant_list = self.merge_bh_list(merging_bh_list)
            self.population.update({bh.id: bh for bh in remnant_list})
            merging_bh_list = remnant_list

        assert self.population_size == BlackHole.bh_counter
        logging.info(f"All BH mergers complete resulting "
                     f"in total number of BH: {self.population_size}")

    @staticmethod
    def __truncate_bh_list(merging_bh_list: List[BlackHole]) -> List[BlackHole]:
        if not len(merging_bh_list) % 2 == 0:
            logging.info(f"Merging BH population uneven, removing a BH")
            merging_bh_list.pop()
        return merging_bh_list



    @staticmethod
    def pair_up_bh(merging_bh_list: List[BlackHole]) -> List[
        Tuple[BlackHole, BlackHole]]:
        # https://stackoverflow.com/questions/5389507/iterating-over-every-two-elements-in-a-list
        paired_up_population = list(zip(merging_bh_list, merging_bh_list[1:]))[::2]
        return paired_up_population

    def merge_bh_list(self, merging_bh_list: List[BlackHole]) -> List[BlackHole]:
        logging.info(f"Merging pairs in list of {len(merging_bh_list)} BH")
        paired_up_population = self.pair_up_bh(merging_bh_list)
        remnants = []
        for bh_1, bh_2 in paired_up_population:
            remnant = self.merge_bbh_pair(bh_1, bh_2)
            remnants.append(remnant)
        return remnants

    def get_generation_counts(self) -> Dict[int, int]:
        generation_counts = dict()
        for bh in self.population.values():
            count = generation_counts.get(bh.generation_number, 0)
            generation_counts.update({bh.generation_number: count + 1})
        return generation_counts

    def get_generation_stats(self) -> pd.DataFrame:
        init_col_vals = [0.0 for _ in range(self.number_of_generation)]
        stats = pd.DataFrame(dict(
            count=[i for i in self.get_generation_counts().values()],
            avg_kick=init_col_vals,
            avg_mass=init_col_vals,
            avg_spin=init_col_vals,
        ))
        # totaling the various stats
        for bh in self.population.values():
            stats.at[bh.generation_number, 'avg_kick'] += utils.mag(bh.kick) * utils.c
            stats.at[bh.generation_number, 'avg_mass'] += bh.mass
            stats.at[bh.generation_number, 'avg_spin'] += utils.mag(bh.spin)
        # averaging the vals
        stats['avg_kick'] = stats['avg_kick'].astype('float') / stats['count'].astype(
            'float')
        stats['avg_mass'] = stats['avg_mass'].astype('float') / stats['count'].astype(
            'float')
        stats['avg_spin'] = stats['avg_spin'].astype('float') / stats['count'].astype(
            'float')

        return stats

    def _get_graph_generation_data(self):
        nodes, edges, labels = [], [], {}
        for bh_id, bh in self.population.items():
            labels.update({bh_id: str(bh)})
            nodes.append(bh_id)
            if bh.parents:
                edges += [(bh.id, p.id) for p in bh.parents]
        edges = utils.remove_duplicate_edges(edges)
        return nodes, edges, labels

    def render_population(self, filename):
        nodes, edges, labels = self._get_graph_generation_data()

        graph = nx.Graph()
        graph.add_nodes_from(nodes)
        graph.add_edges_from(edges)
        fig = plt.figure(figsize=(
            self.number_of_initial_bh * 2,
            self.number_of_generation + 5))
        fig.suptitle(f'{self.number_of_generation - 1} Generations of BH mergers',
                     fontsize=16)
        pos = utils.hierarchy_pos(
            graph=graph,
            root=nodes[-1],
            width=self.number_of_initial_bh * 2,
            vert_gap=0.01
        )
        nx.draw(graph, pos=pos, labels=labels)
        plt.savefig(filename)

    def render_spin_and_mass(self, filename, stats=None):
        if not isinstance(stats, pd.DataFrame):
            stats = self.get_generation_stats()
        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        ax1.overlaid_corner(stats.index, stats.avg_mass, color="red", marker="o")
        ax2.overlaid_corner(stats.index, stats.avg_spin, color="blue", marker="o")
        ax1.set_xlabel("Generation Number", fontsize=14)
        ax1.set_ylabel("Average Mass", color="red", fontsize=14)
        ax2.set_ylabel("Average |\u03C7|", color="blue", fontsize=14)
        ax2.set_ylim(0, 1)
        plt.locator_params(axis="x", integer=True, tight=True)
        plt.margins(0.05)
        plt.tight_layout()
        plt.savefig(filename)

    def run_expiriment(self) -> pd.DataFrame:
        self.__init_population()
        self.conduct_multiple_mergers()
        return self.get_generation_stats()

    def repeat_expirement(self, num_expt: int) -> pd.DataFrame:
        expt_stats = []
        expt_progress_bar = tqdm.tqdm(range(num_expt))
        for expt_num in expt_progress_bar:
            expt_progress_bar.set_description(f"Expt {expt_num}")
            expt_stats.append(self.run_expiriment())
        avg_stats = expt_stats[0]
        for i in range(len(expt_stats)):
            avg_stats = avg_stats.add(expt_stats[i])
        avg_stats = avg_stats / len(expt_stats)
        return avg_stats





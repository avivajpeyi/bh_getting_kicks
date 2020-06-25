import os
import shutil
import unittest
from .black_hole_population import BlackHolePopulation


class TestBlackHolePopulation(unittest.TestCase):

    def setUp(self):
        self.outdir = "test"
        os.makedirs(self.outdir, exist_ok=True)
    #
    # def tearDown(self):
    #     if os.path.exists(self.outdir):
    #         shutil.rmtree(self.outdir)

    def test_black_hole_population(self):
        pop = BlackHolePopulation(number_of_initial_bh=2 ** 6)
        pop.conduct_multiple_mergers()
        gen_counts = pop.get_generation_counts()
        self.assertEqual(gen_counts[0], pop.number_of_initial_bh)
        self.assertEqual(len(gen_counts), pop.number_of_generation)
        fname = os.path.join(self.outdir, f"{pop.number_of_generation-1}_generation_mergers.png")
        pop.render_population(fname)
        self.assertTrue(os.path.isfile(fname))
        fname = os.path.join(self.outdir, f"{pop.number_of_generation-1}_generation_stats.png")
        pop.render_spin_and_mass(fname)
        self.assertTrue(os.path.isfile(fname))
        print(pop.get_generation_stats())

    def test_repeat_expts(self):
        num_expt = 100
        pop = BlackHolePopulation(number_of_initial_bh=2 ** 6)
        avg_stats = pop.repeat_expirement(num_expt=num_expt)
        pop.render_spin_and_mass(
            filename=os.path.join(self.outdir, f"average_of_{num_expt}_simulations.png"),
            stats=avg_stats
        )




if __name__ == '__main__':
    unittest.main()

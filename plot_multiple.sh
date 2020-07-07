#!/bin/bash
#SBATCH --job-name=corner_plotter
#SBATCH --time=2:00:00
#SBATCH --output=corner_plotter.out
#SBATCH --mem-per-cpu=1000

source ~/.bash_profile
conda activate parallel_bilby


python plot_multiple.py
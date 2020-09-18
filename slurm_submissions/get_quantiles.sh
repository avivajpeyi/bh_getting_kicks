#!/bin/bash
#SBATCH --job-name=quantiles_getter
#SBATCH --time=0:05:00
#SBATCH --output=quantiles_getter.out
#SBATCH --mem-per-cpu=1000

source ~/.bash_profile
conda activate parallel_bilby


python get_quantiles_width.py
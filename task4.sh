#!/bin/bash
# Brandon Michaud
#
# disc_dual_a100_students
#SBATCH --partition=debug_5min
#SBATCH --ntasks=1
#SBATCH --cpus-per-task 16
#SBATCH --mem=1G
#SBATCH --output=t-tests.txt
#SBATCH --error=outputs/hw2_%j_stderr.txt
#SBATCH --time=00:05:00
#SBATCH --job-name=hw2
#SBATCH --mail-user=brandondmichaud@ou.edu
#SBATCH --mail-type=ALL
#SBATCH --chdir=/home/cs504319/cs5043-hw2

. /home/fagg/tf_setup.sh
conda activate tf

python task4.py

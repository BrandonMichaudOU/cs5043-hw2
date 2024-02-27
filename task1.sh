#!/bin/bash
# Brandon Michaud
#
# disc_dual_a100_students
#SBATCH --partition=disc_dual_a100_students
#SBATCH --ntasks=1
#SBATCH --cpus-per-task 16
#SBATCH --mem=1G
#SBATCH --output=outputs/hw2_%j_stdout.txt
#SBATCH --error=outputs/hw2_%j_stderr.txt
#SBATCH --time=00:07:00
#SBATCH --job-name=hw2
#SBATCH --mail-user=brandondmichaud@ou.edu
#SBATCH --mail-type=ALL
#SBATCH --chdir=/home/cs504319/cs5043-hw2
# --array=0-69

. /home/fagg/tf_setup.sh
conda activate tf

python hw2_base.py --project 'hw2' --exp_type 'base' --output_type 'ddtheta' --predict_dim 1 --rotation 0 --Ntraining 18 --lrate 0.0001 --activation_hidden 'elu' --activation_out 'linear' --hidden 500 250 125 75 36 17 --epochs 100 -vv

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
#SBATCH --array=17,19,28,32,34,43,50,52,54,56,57,59,62,63,65,66,67,68,69,73,75,76,82,87,90,91,92,93,95,96,97,98,103,104,105,107,109,112,113,115,117,120,125,126,132,133,134,137,142,150,152,154,156,157,160,161,163,164,166,167,168,169,170,173,183,184,185,186,187,188,189,190,192,193,196,199,201,202,203,204,205,207,208,209,220,221,223,226,227,228,229,235,236,237,239,240,241,242,255,256,258,259,260,261,263,264,267,268,272,273,279,305

. /home/fagg/tf_setup.sh
conda activate tf

python hw2_base.py --project 'hw2' --exp_type 'dropout' --output_type 'ddtheta' --predict_dim 1 --exp_index $SLURM_ARRAY_TASK_ID --lrate 0.0001 --activation_hidden 'elu' --activation_out 'linear' --hidden 500 250 125 75 36 17 --epochs 100 -vv
# --min_delta 0.001 --patience 25
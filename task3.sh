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
#SBATCH --array=8,9,11,15,17,18,19,21,22,23,25,26,28,29,30,31,32,34,35,40,47,49,51,52,54,55,58,59,60,61,63,64,67,68,69,71,75,79,80,83,84,85,86,88,91,92,95,98,99,100,101,102,103,104,105,107,110,111,113,115,118,119,120,122,123,125,126,128,130,131,133,134,136,137,138,149,156,159,160,162,163,165,167,169,172,173,176,182,183,185,186,188,190,194,197,200,201,204,205,208,209,214,216,217,218,221,222,223,224,225,226,227,228,229,230,231,234,235,237,238,241,242,243,248,250,251,254,255,260,261,262,264,267,268,271,274,276,277,280,283,285,286,287,289,290,292,293,294,295,299,302,303,304,310,312,315,316,317,321,323,324,349

. /home/fagg/tf_setup.sh
conda activate tf

python hw2_base.py --project 'hw2' --exp_type 'l2' --output_type 'ddtheta' --predict_dim 1 --exp_index $SLURM_ARRAY_TASK_ID --lrate 0.0001 --activation_hidden 'elu' --activation_out 'linear' --hidden 500 250 125 75 36 17 --epochs 100 -vv
# --min_delta 0.001 --patience 25
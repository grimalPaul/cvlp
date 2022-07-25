#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -J filter 
#SBATCH --gres=gpu:1
#SBATCH -p gpu
#SBATCH --mem=40G
#SBATCH --time=0-12:00:00

source /home/pgrimal/.bashrc
source activate cvlp

dataset2=/scratch_global/stage_pgrimal/data/CVLP/data/datasets/wikimage/wikimage_train_val_filter

python -m process_data.print_emptyCLIP \
    --dataset_path=${dataset2}
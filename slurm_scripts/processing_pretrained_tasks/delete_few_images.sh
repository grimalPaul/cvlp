#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -J analyse
#SBATCH --gres=gpu:1
#SBATCH -p gpu
#SBATCH --mem=40G
#SBATCH --time=0-06:00:00

source /home/pgrimal/.bashrc
source activate cvlp

dataset1=
dataset2=

python -m process_data.remove_len2 \
    --dataset_path=${dataset1}

python -m process_data.remove_len2 \
    --dataset_path=${dataset2}
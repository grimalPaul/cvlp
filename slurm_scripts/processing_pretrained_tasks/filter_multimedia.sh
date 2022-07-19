#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -J filter 
#SBATCH --gres=gpu:1
#SBATCH -p gpu
#SBATCH --mem=40G
#SBATCH --time=0-06:00:00

source /home/pgrimal/.bashrc
source activate cvlp

dataset1=/scratch_global/stage_pgrimal/data/CVLP/data/datasets/multimedia/filtered/multimedia_train_val


python -m process_data.remove_len2 \
    --dataset_path=${dataset1}

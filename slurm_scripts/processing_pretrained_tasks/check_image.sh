#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -J check_image
#SBATCH --gres=gpu:1
#SBATCH -p gpu
#SBATCH --mem=5G
#SBATCH --time=0-06:00:00

source /home/pgrimal/.bashrc
source activate cvlp


python -m process_data.check_image
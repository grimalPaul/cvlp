#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -J process_kilt
#SBATCH --gres=gpu:1
#SBATCH -p gpu
#SBATCH --mem=20G

source /home/pgrimal/.bashrc
source activate cvlp

python -m process_data.map_url_images
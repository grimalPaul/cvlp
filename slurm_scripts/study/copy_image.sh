#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -J irrelevant_BM25
#SBATCH --gres=gpu:1
#SBATCH -p gpu
#SBATCH --mem=20G
#SBATCH --time=0-06:00:00

source /home/pgrimal/.bashrc
source activate cvlp

python -m process_data.get_image \
    --image_path=/scratch_global/stage_pgrimal/data/miniViQuAE/data/dataset/miniCommons/ \
    --file_path=/home/pgrimal/CVLEP/results/pure_zero_shot_vlt5/relevant.json \
    --folder_path=/home/pgrimal/CVLEP/results/pure_zero_shot_vlt5/images

echo "DONE"

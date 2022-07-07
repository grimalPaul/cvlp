#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -J embedding_image
#SBATCH --gres=gpu:1
#SBATCH -p gpu
#SBATCH --mem=24G
#SBATCH --time=0-06:00:00

source /home/pgrimal/.bashrc
source activate cvlp

python -m processing.embedding_image \
    --dataset_path=/scratch_global/stage_pgrimal/data/CVLP/data/datasets/wikimage/wikimage_split \
    --type=CLIP \
    --backbone=RN50
    --image_path=/scratch_global/stage_pgrimal/data/CVLP/data/datasets/wikimage/Commons_wikimage \
    --key_image=list_images \
    --key_image_embedding=clip

echo "done"

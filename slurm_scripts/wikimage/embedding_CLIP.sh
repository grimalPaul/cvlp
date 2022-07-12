#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -J clip
#SBATCH --gres=gpu:1
#SBATCH -p gpu
#SBATCH -w node4
#SBATCH --mem=60G
#SBATCH --time=1-00:00:00

source /home/pgrimal/.bashrc
source activate cvlp

python -m processing.embedding_image \
    --dataset_path=/scratch_global/stage_pgrimal/data/CVLP/data/datasets/wikimage/wikimage_no_filter \
    --type=CLIP \
    --backbone=/scratch_global/stage_pgrimal/data/CVLP/data/clip/RN101.pt \
    --image_path=/scratch_global/stage_pgrimal/data/CVLP/data/datasets/wikimage/Commons_wikimage \
    --key_image=list_images \
    --key_image_embedding=clip \
    --batch_size=4 \
    --log_path=/home/pgrimal/CVLEP/error_embedding_clip.json

echo "done"

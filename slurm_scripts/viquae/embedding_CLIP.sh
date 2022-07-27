#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -J clip_viquae
#SBATCH --gres=gpu:1
#SBATCH -p gpu
#SBATCH -w node7
#SBATCH --mem=80G
#SBATCH --time=1-00:00:00

source /home/pgrimal/.bashrc
source activate cvlp

echo "embedding viquae"
python -m processing.embedding_image \
    --dataset_path=/scratch_global/stage_pgrimal/data/CVLP/data/datasets/miniviquae \
    --type=CLIP \
    --backbone=/scratch_global/stage_pgrimal/data/CVLP/data/clip/RN101.pt \
    --image_path=/scratch_global/stage_pgrimal/data/miniViQuAE/data/dataset/miniCommons/ \
    --key_image=image \
    --key_image_embedding=clip \
    --batch_size=4 \
    --log_path=/home/pgrimal/CVLEP/error_embedding_clip_viquae.json

echo "embedding the KB"
python -m processing.embedding_image \
    --dataset_path=/scratch_global/stage_pgrimal/data/CVLP/data/datasets/kb \
    --type=CLIP \
    --backbone=/scratch_global/stage_pgrimal/data/CVLP/data/clip/RN101.pt \
    --image_path=/scratch_global/stage_pgrimal/data/miniViQuAE/data/dataset/miniCommons/ \
    --key_image=image \
    --key_image_embedding=clip \
    --batch_size=4 \
    --log_path=/home/pgrimal/CVLEP/error_embedding_clip_viquae.json

echo "done"
#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -J embedding_question
#SBATCH --gres=gpu:1
#SBATCH -p gpu
#SBATCH --mem=10G
#SBATCH --time=0-06:00:00

source /home/pgrimal/.bashrc
source activate cvlp

python -m processing.embedding_dataset \
    --dataset_path=/scratch_global/stage_pgrimal/data/CVLP/data/datasets/vlt5_viquae_dataset \
    --type=question \
    --model_config_path=experiments/model_cvlep/bergamote/encodersT5.json \
    --key_boxes=vlt5_normalized_boxes \
    --key_vision_features=vlt5_features \
    --key_text=input \
    --key_embedding=vlt5_embedding \
    --batch_size=256

echo "DONE"

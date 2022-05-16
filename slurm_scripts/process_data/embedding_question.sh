#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -J embedding
#SBATCH --gres=gpu:1
#SBATCH -p gpu
#SBATCH --mem=10G
#SBATCH --time=0-06:00:00

source /home/pgrimal/.bashrc
source activate cvlp

python -m ir.embedding_dataset \
    --dataset_path=/scratch_global/stage_pgrimal/data/CVLP/data/datasets/vlt5_viquae_dataset \
    --type=question \
    --model_config_path=experiments/model_cvlep/bergamote/encodersT5.json \
    --key_boxes=vlt5_normalized_boxes \
    --key_vision_features=vlt5_features \
    --key_token=vlt5_input_ids \
    --key_embedding=vlt5_embedding
    
echo "DONE"

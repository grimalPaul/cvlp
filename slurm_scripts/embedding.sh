#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -J embedding
#SBATCH --gres=gpu:1
#SBATCH -p gpu
#SBATCH --mem=4G
#SBATCH --time=0-06:00:00

source /home/pgrimal/.bashrc
source activate cvlp

python -m ir.embedding_dataset \
    --path_dataset=/scratch_global/stage_pgrimal/data/miniViQuAE/data/wikidata_id/vlt5/vlt5_viquae_dataset \
    --type=question \
    --path_config_model=experiments/model_cvlep/bergamote/encodersT5.json \
    --key_boxes=vlt5_normalized_boxes \
    --key_vision_features=vlt5_normalized_boxes \
    --key_token=vlt5_input_id \
    --key_embedding=vlt5_embedding


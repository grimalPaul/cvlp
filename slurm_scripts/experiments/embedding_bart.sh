#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -J embedding_passage
#SBATCH --gres=gpu:1
#SBATCH -p gpu
#SBATCH -w node4
#SBATCH --mem=30G
#SBATCH --time=7-00:00:00


source /home/pgrimal/.bashrc
source activate cvlp

echo "dataset"
python -m processing.embedding_dataset \
    --dataset_path=/scratch_global/stage_pgrimal/data/CVLP/data/datasets/vlbart_viquae_dataset \
    --type=question \
    --model_config_path=experiments/model_cvlep/bergamote/encodersBart.json \
    --key_boxes=vlt5_normalized_boxes \
    --key_vision_features=vlt5_features \
    --key_text=input \
    --key_embedding=vlbart_embedding_1token \
    --kb_path=/scratch_global/stage_pgrimal/data/CVLP/data/datasets/kb \
    --batch_size=64 \
    --pool_strategy=1token
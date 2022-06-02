#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -J embedding
#SBATCH --gres=gpu:1
#SBATCH -p gpu
#SBATCH --mem=20G
#SBATCH --time=7-00:00:00

source /home/pgrimal/.bashrc
source activate cvlp

echo "vlt5_zs_avg_dict"

echo "dataset avg"

python -m processing.embedding_dataset \
    --dataset_path=/scratch_global/stage_pgrimal/data/CVLP/data/datasets/vlt5_viquae_dataset \
    --type=question \
    --config_question_path=experiments/configEncoder/bergamote/args_VLT5.json \
    --config_passage_path=experiments/configEncoder/bergamote/args_VLT5.json \
    --config_training_path=experiments/configEncoder/training_params/training_vlt5.json \
    --key_boxes=vlt5_normalized_boxes \
    --key_vision_features=vlt5_features \
    --key_text=input \
    --key_embedding=vlt5_embedding_avg_dict \
    --kb_path=/scratch_global/stage_pgrimal/data/CVLP/data/datasets/kb \
    --batch_size=128 \
    --pool_strategy=avg

echo "passages avg"
python -m processing.embedding_dataset \
    --dataset_path=/scratch_global/stage_pgrimal/data/CVLP/data/datasets/passages \
    --type=passage \
    --config_question_path=experiments/configEncoder/bergamote/args_VLT5.json \
    --config_passage_path=experiments/configEncoder/bergamote/args_VLT5.json \
    --config_training_path=experiments/configEncoder/training_params/training_vlt5.json \
    --key_boxes=vlt5_normalized_boxes \
    --key_vision_features=vlt5_features \
    --key_text=passage \
    --key_embedding=vlt5_embedding_avg_dict \
    --kb_path=/scratch_global/stage_pgrimal/data/CVLP/data/datasets/kb \
    --batch_size=128 \
    --pool_strategy=avg

echo "vlt5 zs avg"
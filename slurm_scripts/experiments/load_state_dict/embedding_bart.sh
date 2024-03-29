#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -J embed_bart
#SBATCH --gres=gpu:1
#SBATCH -p gpu
#SBATCH --mem=40G
#SBATCH --time=7-00:00:00

source /home/pgrimal/.bashrc
source activate cvlp

echo "vlbart_zs_avg"
echo "dataset"
python -m processing.embedding_dataset \
    --dataset_path=/scratch_global/stage_pgrimal/data/CVLP/data/datasets/vlbart_viquae_dataset \
    --type=question \
    --config_question_path=experiments/configEncoder/bergamote/args_bart.json \
    --config_passage_path=experiments/configEncoder/bergamote/args_bart.json \
    --config_training_path=experiments/configEncoder/training_params/training_vlbart.json \
    --key_boxes=vlt5_normalized_boxes \
    --key_vision_features=vlt5_features \
    --key_text=input \
    --key_embedding=vlbart_embedding_avg_dict \
    --kb_path=/scratch_global/stage_pgrimal/data/CVLP/data/datasets/kb \
    --batch_size=128 \
    --pool_strategy=avg

echo "passages"
python -m processing.embedding_dataset \
    --dataset_path=/scratch_global/stage_pgrimal/data/CVLP/data/datasets/bart_passages \
    --type=passage \
    --config_question_path=experiments/configEncoder/bergamote/args_bart.json \
    --config_passage_path=experiments/configEncoder/bergamote/args_bart.json \
    --config_training_path=experiments/configEncoder/training_params/training_vlbart.json \
    --key_boxes=vlt5_normalized_boxes \
    --key_vision_features=vlt5_features \
    --key_text=passage \
    --key_embedding=vlbart_embedding_avg_dict \
    --kb_path=/scratch_global/stage_pgrimal/data/CVLP/data/datasets/kb \
    --batch_size=64 \
    --pool_strategy=avg

echo "DONE"
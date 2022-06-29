#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -J embedding_Q
#SBATCH --gres=gpu:1
#SBATCH -p gpu
#SBATCH -w node7
#SBATCH --mem=10G
#SBATCH --time=7-00:00:00

source /home/pgrimal/.bashrc
source activate cvlp

# variables
dataset=/scratch_global/stage_pgrimal/data/CVLP/data/datasets/vlt5_dataset
kb=/scratch_global/stage_pgrimal/data/CVLP/data/datasets/kb
passages=/scratch_global/stage_pgrimal/data/CVLP/data/datasets/vlt5_passages
config_passage_path=experiments/config_vladapter/bergamote/trained_models/adapter_projection/passage_encoder_adapter_projection.json
config_question_path=experiments/config_vladapter/bergamote/trained_models/adapter_projection/question_encoder_adapter_projection.json
config_model_path=experiments/config_vladapter/bergamote/trained_models/adapter_projection/config_model.json
batch_size=64

echo "Dataset"

python -m processing.embedding_dataset \
    --dataset_path=${dataset} \
    --type=question \
    --config_question_path=${config_question_path} \
    --config_passage_path=${config_passage_path} \
    --config_model_path=${config_model_path} \
    --key_boxes=vlt5_normalized_boxes \
    --key_vision_features=vlt5_features \
    --key_text=input \
    --key_embedding=adapter_projection_embedding \
    --batch_size=${batch_size}

echo "Done"
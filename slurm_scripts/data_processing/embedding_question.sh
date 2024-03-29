#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -J embed_question
#SBATCH --gres=gpu:1
#SBATCH -p gpu
#SBATCH -w node4
#SBATCH --mem=10G
#SBATCH --time=7-00:00:00

source /home/pgrimal/.bashrc
source activate cvlp

# variables
dataset=/scratch_global/stage_pgrimal/data/CVLP/data/datasets/vlt5_dataset
kb=/scratch_global/stage_pgrimal/data/CVLP/data/datasets/kb
passages=/scratch_global/stage_pgrimal/data/CVLP/data/datasets/vlt5_passages
config_passage_path=experiments/config_vladapter/bergamote/trained_models/prompt/passage_encoder_prompting.json
config_question_path=experiments/config_vladapter/bergamote/trained_models/prompt/question_encoder_prompting.json
config_model_path=experiments/config_vladapter/bergamote/trained_models/prompt/config_model.json
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
    --key_embedding=prompt_tuning_embedding \
    --batch_size=${batch_size}

echo "Done"
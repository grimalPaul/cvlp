#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -J embed_question
#SBATCH --gres=gpu:1
#SBATCH -p gpu
#SBATCH -w node6
#SBATCH --mem=10G
#SBATCH --time=0-10:00:00

source /home/pgrimal/.bashrc
source activate cvlp

# variables
dataset=/scratch_global/stage_pgrimal/data/CVLP/data/datasets/miniviquae
kb=/scratch_global/stage_pgrimal/data/CVLP/data/datasets/kb
passages=/scratch_global/stage_pgrimal/data/CVLP/data/datasets/passages

config_passage_path=experiments/config_vladapter/bergamote/embedding/encoder_passage.json
config_question_path=experiments/config_vladapter/bergamote/embedding/encoder_question.json
config_model_path=experiments/config_vladapter/bergamote/embedding/config_model.json
batch_size=64
key_embedding=multitask_resnet_embedding

echo "Dataset"

python -m processing.embedding_dataset \
    --dataset_path=${dataset} \
    --type=question \
    --config_question_path=${config_question_path} \
    --config_passage_path=${config_passage_path} \
    --config_model_path=${config_model_path} \
    --key_boxes=fastrcnn_boxes \
    --key_vision_features=fastrcnn_features \
    --key_text=input \
    --key_embedding=${key_embedding} \
    --batch_size=${batch_size}

echo "Done"

echo "Done"
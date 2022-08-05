#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -J embed
#SBATCH --gres=gpu:1
#SBATCH -w node4
#SBATCH --mem=40G
#SBATCH --time=0-12:00:00

source /home/pgrimal/.bashrc
source activate cvlp

# variables
dataset=/scratch_global/stage_pgrimal/data/CVLP/data/datasets/zero_and_finetuning_test/vlt5_dataset
kb=/scratch_global/stage_pgrimal/data/CVLP/data/datasets/zero_and_finetuning_test/kb
passages=/scratch_global/stage_pgrimal/data/CVLP/data/datasets/zero_and_finetuning_test/passages

config_passage_path=experiments/config_vladapter/bergamote/embedding/encoder_passage.json
config_question_path=experiments/config_vladapter/bergamote/embedding/encoder_question.json
config_model_path=experiments/config_vladapter/bergamote/embedding/config_model.json
batch_size=128
key_embedding=multitask_fasterrcnn_embedding


echo "Passage"
python -m processing.embedding_dataset \
    --dataset_path=${passages} \
    --type=passage \
    --config_question_path=${config_question_path} \
    --config_passage_path=${config_passage_path} \
    --config_model_path=${config_model_path} \
    --key_boxes=vlt5_normalized_boxes \
    --key_vision_features=vlt5_features \
    --key_text=passage \
    --key_embedding=${key_embedding} \
    --kb_path=${kb} \
    --batch_size=${batch_size}


echo "dataset"
python -m processing.embedding_dataset \
    --dataset_path=${dataset} \
    --type=question \
    --config_question_path=${config_question_path} \
    --config_passage_path=${config_passage_path} \
    --config_model_path=${config_model_path} \
    --key_boxes=vlt5_normalized_boxes \
    --key_vision_features=vlt5_features \
    --key_text=input \
    --key_embedding=${key_embedding} \
    --batch_size=${batch_size}

echo "Done"
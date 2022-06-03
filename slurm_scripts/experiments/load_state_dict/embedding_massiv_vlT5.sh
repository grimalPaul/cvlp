#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -J embedding
#SBATCH --gres=gpu:1
#SBATCH -p gpu
#SBATCH -w node4
#SBATCH --mem=20G
#SBATCH --time=7-00:00:00

source /home/pgrimal/.bashrc
source activate cvlp

echo "vlt5"

# variables
dataset=/scratch_global/stage_pgrimal/data/CVLP/data/datasets/vlt5_dataset
kb=/scratch_global/stage_pgrimal/data/CVLP/data/datasets/kb
passages=/scratch_global/stage_pgrimal/data/CVLP/data/datasets/vlt5_passages
config_passage_path=experiments/configEncoder/bergamote/args_VLT5.json
config_question_path=experiments/configEncoder/bergamote/args_VLT5.json
config_training_path=experiments/configEncoder/training_params/training_vlt5.json
batch_size=128

echo "------without prefix-------"
echo "1token avec pad et sep"
echo "dataset"
python -m processing.embedding_dataset \
    --dataset_path=${dataset} \
    --type=question \
    --config_question_path=${config_question_path} \
    --config_passage_path=${config_passage_path} \
    --config_training_path=${config_training_path} \
    --key_boxes=vlt5_normalized_boxes \
    --key_vision_features=vlt5_features \
    --key_text=input_pad \
    --key_embedding=vlt5_embedding_1token \
    --kb_path=${kb} \
    --batch_size=${batch_size} \
    --pool_strategy=1token

echo "passages"
python -m processing.embedding_dataset \
    --dataset_path=${passages} \
    --type=passage \
    --config_question_path=${config_question_path} \
    --config_passage_path=${config_passage_path} \
    --config_training_path=${config_training_path} \
    --key_boxes=vlt5_normalized_boxes \
    --key_vision_features=vlt5_features \
    --key_text=passage_pad \
    --key_embedding=vlt5_embedding_1token \
    --kb_path=${kb} \
    --batch_size=${batch_size} \
    --pool_strategy=1token

echo "avg avec sep"
echo "dataset"
python -m processing.embedding_dataset \
    --dataset_path=${dataset} \
    --type=question \
    --config_question_path=${config_question_path} \
    --config_passage_path=${config_passage_path} \
    --config_training_path=${config_training_path} \
    --key_boxes=vlt5_normalized_boxes \
    --key_vision_features=vlt5_features \
    --key_text=input \
    --key_embedding=vlt5_embedding_avg \
    --kb_path=${kb} \
    --batch_size=${batch_size} \
    --pool_strategy=avg

echo "passages"
python -m processing.embedding_dataset \
    --dataset_path=${passages} \
    --type=passage \
    --config_question_path=${config_question_path} \
    --config_passage_path=${config_passage_path} \
    --config_training_path=${config_training_path} \
    --key_boxes=vlt5_normalized_boxes \
    --key_vision_features=vlt5_features \
    --key_text=passage \
    --key_embedding=vlt5_embedding_avg \
    --kb_path=${kb} \
    --batch_size=${batch_size} \
    --pool_strategy=avg

echo "----------imt-----------"

echo "dataset sep et pad 1token"

python -m processing.embedding_dataset \
    --dataset_path=${dataset} \
    --type=question \
    --config_question_path=${config_question_path} \
    --config_passage_path=${config_passage_path} \
    --config_training_path=${config_training_path} \
    --key_boxes=vlt5_normalized_boxes \
    --key_vision_features=vlt5_features \
    --key_text=input_imt_pad \
    --key_embedding=vlt5_embedding_imt_1token \
    --kb_path=${kb} \
    --batch_size=${batch_size} \
    --pool_strategy=1token

echo "passages sep et pad 1token"

python -m processing.embedding_dataset \
    --dataset_path=${passages} \
    --type=question \
    --config_question_path=${config_question_path} \
    --config_passage_path=${config_passage_path} \
    --config_training_path=${config_training_path} \
    --key_boxes=vlt5_normalized_boxes \
    --key_vision_features=vlt5_features \
    --key_text=passage_imt_pad \
    --key_embedding=vlt5_embedding_imt_1token \
    --kb_path=${kb} \
    --batch_size=${batch_size} \
    --pool_strategy=1token

echo "dataset sep avg"

python -m processing.embedding_dataset \
    --dataset_path=${dataset} \
    --type=question \
    --config_question_path=${config_question_path} \
    --config_passage_path=${config_passage_path} \
    --config_training_path=${config_training_path} \
    --key_boxes=vlt5_normalized_boxes \
    --key_vision_features=vlt5_features \
    --key_text=input_imt \
    --key_embedding=vlt5_embedding_imt_avg \
    --kb_path=${kb} \
    --batch_size=${batch_size} \
    --pool_strategy=avg

echo "passages sep avg"

python -m processing.embedding_dataset \
    --dataset_path=${passages} \
    --type=question \
    --config_question_path=${config_question_path} \
    --config_passage_path=${config_passage_path} \
    --config_training_path=${config_training_path} \
    --key_boxes=vlt5_normalized_boxes \
    --key_vision_features=vlt5_features \
    --key_text=passage_imt \
    --key_embedding=vlt5_embedding_imt_avg \
    --kb_path=${kb} \
    --batch_size=${batch_size} \
    --pool_strategy=avg

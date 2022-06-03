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

echo "vlbart"

# variables
dataset=/scratch_global/stage_pgrimal/data/CVLP/data/datasets/vlbart_dataset
kb=/scratch_global/stage_pgrimal/data/CVLP/data/datasets/kb
passages=/scratch_global/stage_pgrimal/data/CVLP/data/datasets/vlbart_passages
config_passage_path=experiments/configEncoder/bergamote/args_bart.json
config_question_path=experiments/configEncoder/bergamote/args_bart.json
config_training_path=experiments/configEncoder/training_params/training_vlbart.json
batch_size=64

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
    --key_text=input_cls \
    --key_embedding=vlbart_embedding_1token \
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
    --key_text=passage_sep_cls \
    --key_embedding=vlbart_embedding_1token \
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
    --key_embedding=vlbart_embedding_avg \
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
    --key_text=passage_sep \
    --key_embedding=vlbart_embedding_avg \
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
    --key_text=input_imt_cls \
    --key_embedding=vlbart_embedding_imt_1token \
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
    --key_text=passage_imt_sep_cls \
    --key_embedding=vlbart_embedding_imt_1token \
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
    --key_embedding=vlbart_embedding_imt_avg \
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
    --key_text=passage_imt_sep \
    --key_embedding=vlbart_embedding_imt_avg \
    --kb_path=${kb} \
    --batch_size=${batch_size} \
    --pool_strategy=avg
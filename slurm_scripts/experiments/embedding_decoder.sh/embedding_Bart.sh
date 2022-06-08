#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -J embed_bart
#SBATCH --gres=gpu:1
#SBATCH -p gpu
#SBATCH -w node4
#SBATCH --mem=10G
#SBATCH --time=7-00:00:00

source /home/pgrimal/.bashrc
source activate cvlp

# variables
dataset=/scratch_global/stage_pgrimal/data/CVLP/data/datasets/vlbart_dataset
kb=/scratch_global/stage_pgrimal/data/CVLP/data/datasets/kb
passages=/scratch_global/stage_pgrimal/data/CVLP/data/datasets/vlbart_passages
config_passage_path=experiments/configEncoder/bergamote/args_bart.json
config_question_path=experiments/configEncoder/bergamote/args_bart.json
config_training_path=experiments/configEncoder/training_params/training_vlbart.json
batch_size=64

echo "Decoder BART"
echo "-----without prefix----"
echo "Dataset"
<<com
python -m processing.embedding_dataset \
    --dataset_path=${dataset} \
    --type=question \
    --config_question_path=${config_question_path} \
    --config_passage_path=${config_passage_path} \
    --config_training_path=${config_training_path} \
    --key_boxes=vlt5_normalized_boxes \
    --key_vision_features=vlt5_features \
    --key_text=input \
    --key_embedding=vlbart_decoder \
    --batch_size=${batch_size}

com
echo "Passage"
python -m processing.embedding_dataset \
    --dataset_path=${passages} \
    --type=passage \
    --config_question_path=${config_question_path} \
    --config_passage_path=${config_passage_path} \
    --config_training_path=${config_training_path} \
    --key_boxes=vlt5_normalized_boxes \
    --key_vision_features=vlt5_features \
    --key_text=passage_sep \
    --key_embedding=vlbart_decoder \
    --kb_path=${kb} \
    --batch_size=${batch_size}

echo "----With imt----"
echo "Dataset"
<<com
python -m processing.embedding_dataset \
    --dataset_path=${dataset} \
    --type=question \
    --config_question_path=${config_question_path} \
    --config_passage_path=${config_passage_path} \
    --config_training_path=${config_training_path} \
    --key_boxes=vlt5_normalized_boxes \
    --key_vision_features=vlt5_features \
    --key_text=input_imt \
    --key_embedding=vlbart_imt_decoder \
    --batch_size=${batch_size}

com
echo "Passage"
python -m processing.embedding_dataset \
    --dataset_path=${passages} \
    --type=passage \
    --config_question_path=${config_question_path} \
    --config_passage_path=${config_passage_path} \
    --config_training_path=${config_training_path} \
    --key_boxes=vlt5_normalized_boxes \
    --key_vision_features=vlt5_features \
    --key_text=passage_imt_sep \
    --key_embedding=vlbart_imt_decoder \
    --kb_path=${kb} \
    --batch_size=${batch_size}

echo "Done"
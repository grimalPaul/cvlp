#!/usr/bin/env sh
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH -J embed
#SBATCH --gres=gpu:1
#SBATCH --partition=gpuv100
#SBATCH --mem=20G
#SBATCH --time=0-10:00:00
#SBATCH -w node27

source /home/users/pgrimal/.bashrc
source activate cvlp

# variables
dataset=/home/users/pgrimal/data/datasets/cvlp/miniviquae
kb=/home/users/pgrimal/data/datasets/cvlp/kb
passages=/home/users/pgrimal/data/datasets/cvlp/passages

config_passage_path=experiments/config_vladapter/factoryIA/embedding/encoder_passage.json
config_question_path=experiments/config_vladapter/factoryIA/embedding/encoder_question.json
config_model_path=experiments/config_vladapter/factoryIA/embedding/config_model.json
batch_size=256
key_embedding=finetuning_clip_embedding

echo "Dataset"

python -m processing.embedding_dataset \
    --dataset_path=${dataset} \
    --type=question \
    --config_question_path=${config_question_path} \
    --config_passage_path=${config_passage_path} \
    --config_model_path=${config_model_path} \
    --key_vision_features=clip_features \
    --key_text=input \
    --key_embedding=${key_embedding} \
    --batch_size=${batch_size}


echo "Passage"
python -m processing.embedding_dataset \
    --dataset_path=${passages} \
    --type=passage \
    --config_question_path=${config_question_path} \
    --config_passage_path=${config_passage_path} \
    --config_model_path=${config_model_path} \
    --key_vision_features=clip_features \
    --key_text=passage \
    --key_embedding=${key_embedding} \
    --kb_path=${kb} \
    --batch_size=${batch_size}

echo "Done"
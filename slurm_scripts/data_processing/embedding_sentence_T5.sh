#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -J embed
#SBATCH --gres=gpu:1
#SBATCH -w node4
#SBATCH --mem=40G
#SBATCH --time=0-12:00:00

source /home/pgrimal/.bashrc
source activate sentenceT5

# variables
dataset=/scratch_global/stage_pgrimal/data/CVLP/data/datasets/miniviquae
passages=/scratch_global/stage_pgrimal/data/CVLP/data/datasets/passages
model_path=/scratch_global/stage_pgrimal/data/CVLP/data/SentenceTransformer/sentence-T5
batch_size=256
key_embedding=sentence_T5

echo "Passage"
python -m processing.embedding_sentence_T5_only \
    --batch_size=${batch_size} \
    --model_path=${model_path} \
    --key_text=passage \
    --key_embedding=${key_embedding} \
    --dataset_path=${passages}

echo "dataset"
python -m processing.embedding_sentence_T5_only \
    --batch_size=${batch_size} \
    --model_path=${model_path} \
    --key_text=input \
    --key_embedding=${key_embedding} \
    --dataset_path=${dataset}

source /home/pgrimal/.bashrc
source activate cvlp

echo "research"

python -m search \
    --dataset_path=/scratch_global/stage_pgrimal/data/CVLP/data/datasets/miniviquae/test \
    --config=experiments/ir/sentence-T5/sentence_T5.json \
    --metrics_path=experiments/ir/sentence-T5 \
    --k=100 \
    --batch_size=128

echo "Done"
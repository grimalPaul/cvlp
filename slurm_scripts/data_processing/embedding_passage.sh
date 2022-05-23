#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -J embedding_passage
#SBATCH --gres=gpu:1
#SBATCH -p gpu
#SBATCH -w node7
#SBATCH --mem=10G
#SBATCH --time=0-06:00:00

source /home/pgrimal/.bashrc
source activate cvlp

python -m processing.embedding_dataset \
    --dataset_path=/scratch_global/stage_pgrimal/data/CVLP/data/datasets/passages \
    --type=passage \
    --model_config_path=experiments/model_cvlep/bergamote/encodersT5.json \
    --key_boxes=vlt5_normalized_boxes \
    --key_vision_features=vlt5_features \
    --key_text=passage \
    --key_embedding=vlt5_embedding \
    --kb_path=/scratch_global/stage_pgrimal/data/CVLP/data/datasets/kb \
    --batch_size=256
echo "DONE"

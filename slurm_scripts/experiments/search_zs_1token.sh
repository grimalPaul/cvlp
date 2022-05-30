#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -J search
#SBATCH --gres=gpu:1
#SBATCH -p gpu
#SBATCH --mem=20G
#SBATCH --time=0-06:00:00

source /home/pgrimal/.bashrc
source activate cvlp

echo "T5"
python -m search \
    --dataset_path=/scratch_global/stage_pgrimal/data/CVLP/data/datasets/vlt5_viquae_dataset/test \
    --config=experiments/ir/VL/experiments/zs_1token_T5.json \
    --metrics_path=experiments/ir/VL/experiments/zs_T5/ \
    --k=100 \
    --batch_size=256

echo "BART"
python -m search \
    --dataset_path=/scratch_global/stage_pgrimal/data/CVLP/data/datasets/vlbart_viquae_dataset/test \
    --config=experiments/ir/VL/experiments/zs_1token_bart.json \
    --metrics_path=experiments/ir/VL/experiments/zs_bart/ \
    --k=100 \
    --batch_size=256

echo "DONE"

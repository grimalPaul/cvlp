#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -J search
#SBATCH --gres=gpu:1
#SBATCH -p gpu
#SBATCH --mem=20G
#SBATCH --time=0-06:00:00
#SBATCH --dependency=117296
source /home/pgrimal/.bashrc
source activate cvlp

echo "fastrcnn"

python -m search \
    --dataset_path=/scratch_global/stage_pgrimal/data/CVLP/data/datasets/zero_and_finetuning_test/vlt5_dataset/test \
    --config=experiments/ir/VL/experiments/faster_rcnn_multitask/multitask.json \
    --metrics_path=experiments/ir/VL/experiments/VLT5_pretrained \
    --k=100 \
    --batch_size=64

echo "DONE"
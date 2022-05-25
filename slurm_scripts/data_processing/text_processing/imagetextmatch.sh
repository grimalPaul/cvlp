#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -J irrelevant_BM25
#SBATCH --gres=gpu:1
#SBATCH -p gpu
#SBATCH --mem=20G
#SBATCH --time=0-06:00:00

source /home/pgrimal/.bashrc
source activate cvlp

echo "dataset"

python -m processing.text_processing \
    --key=input \
    --dataset_path=/scratch_global/stage_pgrimal/data/CVLP/data/datasets/vlt5_viquae_dataset \
    --prefix='image text match'\
    --name=imt

echo "kb"

python -m processing.text_processing \
    --key=passage \
    --dataset_path=/scratch_global/stage_pgrimal/data/CVLP/data/datasets/passages \
    --prefix='image text match'\
    --name=imt

echo "DONE"

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

python -m processing.analyse_result \
    --key=vlt5_test1 \
    --dataset_path=/scratch_global/stage_pgrimal/data/CVLP/data/datasets/vlt5_viquae_dataset/test \
    --kb_path=/scratch_global/stage_pgrimal/data/CVLP/data/datasets/kb \
    --passages_path=/scratch_global/stage_pgrimal/data/CVLP/data/datasets/passages \
    --k=3 \
    --save_path=/home/pgrimal/CVLEP/results

echo "DONE"

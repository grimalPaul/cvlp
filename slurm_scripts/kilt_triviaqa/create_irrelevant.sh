#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -J irrelevant_BM25
#SBATCH --gres=gpu:1
#SBATCH -p gpu
#SBATCH --mem=50G
#SBATCH --time=1-00:00:00

source /home/pgrimal/.bashrc
source activate cvlp

python -m processing.irrelevant \
    --indice=BM25 \
    --passages_path=/scratch_global/stage_pgrimal/data/CVLP/data/datasets/kilt/passages \
    --dataset_path=/scratch_global/stage_pgrimal/data/CVLP/data/datasets/kilt/triviaqa_for_viquae

echo "DONE"

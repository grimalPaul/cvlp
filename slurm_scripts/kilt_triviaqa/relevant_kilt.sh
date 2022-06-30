#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -J process_kilt
#SBATCH --gres=gpu:1
#SBATCH -p gpu
#SBATCH -w node5
#SBATCH --mem=90G

source /home/pgrimal/.bashrc
source activate cvlp

echo "search relevant"

python -m meerqat.ir.metrics relevant \
    /scratch_global/stage_pgrimal/data/CVLP/data/datasets/kilt/triviaqa_for_viquae \
    /scratch_global/stage_pgrimal/data/CVLP/data/datasets/kilt/passages \
    /scratch_global/stage_pgrimal/data/CVLP/data/datasets/kilt/title2index.json \
    /scratch_global/stage_pgrimal/data/CVLP/data/datasets/kilt/article2passage.json

echo "Done"
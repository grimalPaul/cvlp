#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -J relevant
#SBATCH --gres=gpu:1
#SBATCH -p gpu

echo "Relevant"

source /home/pgrimal/.bashrc
source activate cvlp

python -m meerqat.ir.metrics relevant /scratch_global/stage_pgrimal/data/CVLP/data/datasets/viquae_dataset \
    /scratch_global/stage_pgrimal/data/CVLP/data/datasets/passages \
    /scratch_global/stage_pgrimal/data/CVLP/data/datasets/title2index.json \
    /scratch_global/stage_pgrimal/data/CVLP/data/datasets/article2passage.json

echo "Done"

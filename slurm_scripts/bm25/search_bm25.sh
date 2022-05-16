#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -J bm25search
#SBATCH --gres=gpu:1
#SBATCH -p gpu
#SBATCH -w node4

source /home/pgrimal/.bashrc
source activate cvlp

echo 'BM25 search'
python -m meerqat.ir.search \
    /scratch_global/stage_pgrimal/data/CVLP/data/datasets/vlt5_viquae_dataset \
    experiments/ir/viquae/bm25/config.json --k=100 \
    --metrics=experiments/ir/viquae/bm25/metrics

echo "Done"
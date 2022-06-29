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
    /scratch_global/stage_pgrimal/data/CVLP/data/datasets/kilt/triviaqa_for_viquae \
    experiments/ir/triviaqa_for_viquae/config.json \
    --k=100 \
    --metrics=experiments/ir/triviaqa_for_viquae/metrics

echo "Done"

#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -J bm25opt
#SBATCH --gres=gpu:1
#SBATCH -p gpu
#SBATCH -w node4

source /home/pgrimal/.bashrc
source activate cvlp

echo 'optimize BM25'
python -m meerqat.ir.hp bm25 /scratch_global/stage_pgrimal/data/CVLP/data/datasets/vlt5_viquae_dataset/validation \
    experiments/ir/viquae/hp/bm25/config.json --k=100 \
    --test=/scratch_global/stage_pgrimal/data/CVLP/data/datasets/vlt5_viquae_dataset/test \
    --metrics=experiments/ir/viquae/hp/bm25/metrics

echo "Done"

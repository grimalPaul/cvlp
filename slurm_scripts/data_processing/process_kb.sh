#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -J process_kb
#SBATCH --gres=gpu:1
#SBATCH -p gpu

source /home/pgrimal/.bashrc
source activate cvlp

echo 'create passage dataset'
python -m meerqat.data.loading passages /scratch_global/stage_pgrimal/data/CVLP/data/datasets/kb /scratch_global/stage_pgrimal/data/CVLP/data/datasets/passages experiments/passages/config.json

echo 'title2index'
python -m meerqat.data.loading map /scratch_global/stage_pgrimal/data/CVLP/data/datasets/kb wikipedia_title /scratch_global/stage_pgrimal/data/CVLP/data/datasets/title2index.json --inverse

echo 'article2passage'
python -m meerqat.data.loading map /scratch_global/stage_pgrimal/data/CVLP/data/datasets/kb passage_index /scratch_global/stage_pgrimal/data/CVLP/data/datasets/article2passage.json

echo 'done'
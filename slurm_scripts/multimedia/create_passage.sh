#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -J process_kilt
#SBATCH --gres=gpu:1
#SBATCH -p gpu
#SBATCH --mem=90G

source /home/pgrimal/.bashrc
source activate cvlp

path_kb=/scratch_global/stage_pgrimal/data/CVLP/data/datasets/multimedia/multimedia

echo 'create passage dataset'
python -m meerqat.data.loading passages \
    ${path_kb} \
    /scratch_global/stage_pgrimal/data/CVLP/data/datasets/multimedia/passages \
    experiments/passages/config.json

echo 'title2index'
python -m meerqat.data.loading map \
    ${path_kb} \
    wikipedia_title \
    /scratch_global/stage_pgrimal/data/CVLP/data/datasets/multimedia/title2index.json \
    --inverse

echo 'article2passage'
python -m meerqat.data.loading map \
    ${path_kb} \
    passage_index \
    /scratch_global/stage_pgrimal/data/CVLP/data/datasets/multimedia/article2passage.json

echo 'done'
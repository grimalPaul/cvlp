#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -J process_kilt
#SBATCH --gres=gpu:1
#SBATCH -p gpu
#SBATCH --mem=120G

source /home/pgrimal/.bashrc
source activate cvlp

echo 'create passage dataset'
python -m meerqat.data.loading passages \
    /scratch_global/stage_pgrimal/data/CVLP/data/datasets/kilt/kilt_trivia \
    /scratch_global/stage_pgrimal/data/CVLP/data/datasets/kilt/passages \
    experiments/passages/config.json
<<com

echo 'title2index'
python -m meerqat.data.loading map \
    /scratch_global/stage_pgrimal/data/CVLP/data/datasets/kilt/kilt_trivia \
    wikipedia_title \
    /scratch_global/stage_pgrimal/data/CVLP/data/datasets/kilt/title2index.json \
    --inverse

com

echo 'article2passage'
python -m meerqat.data.loading map \
    /scratch_global/stage_pgrimal/data/CVLP/data/datasets/kilt/kilt_trivia \
    passage_index \
    /scratch_global/stage_pgrimal/data/CVLP/data/datasets/kilt/article2passage.json


echo 'done'
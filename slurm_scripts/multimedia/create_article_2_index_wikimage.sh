#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -J process_kilt
#SBATCH --gres=gpu:1
#SBATCH -p gpu
#SBATCH --mem=40G

source /home/pgrimal/.bashrc
source activate cvlp

path_kb=/scratch_global/stage_pgrimal/data/CVLP/data/datasets/wikimage/wikimage_no_filter

echo 'title2index'
python -m meerqat.data.loading map \
    ${path_kb} \
    wikipedia_title \
   /scratch_global/stage_pgrimal/data/CVLP/data/datasets/wikimage/title2index.json \
    --inverse

echo 'done'
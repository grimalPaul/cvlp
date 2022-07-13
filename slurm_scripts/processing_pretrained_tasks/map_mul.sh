#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -J map_mul
#SBATCH -p gpu
#SBATCH --mem=50G
#SBATCH --time=2-00:00:00
#SBATCH --cpus-per-task=4

source /home/pgrimal/.bashrc
source activate cvlp

dataset1=/scratch_global/stage_pgrimal/data/CVLP/data/datasets/wikimage/wikimage_split_v2
dataset2=/scratch_global/stage_pgrimal/data/CVLP/data/datasets/multimedia/filtered/multimedia_split_v2
wikimage=/scratch_global/stage_pgrimal/data/CVLP/data/datasets/wikimage/wikimage_no_filter/
title2index=/scratch_global/stage_pgrimal/data/CVLP/data/datasets/wikimage/title2index.json
key_b_FN=fastrcnn_boxes
key_f_F=fastrcnn_features
key_CLIP=clip_features

echo 'map embed'

echo 'multi embed'
python -m process_data.map_embedding \
    --dataset_path=${dataset2} \
    --key_features_FasterRCNN=${key_f_F} \
    --key_boxes_FasterRCNN=${key_b_FN} \
    --key_featuresCLIP=${key_CLIP} \
    --wikimage_path=${wikimage} \
    --title2index_wikimage=${title2index} \
<<com
echo 'wiki rmv'
python -m process_data.remove_len2 \
    --dataset_path=${dataset1}

echo 'mul rmv'
python -m process_data.remove_len2 \
    --dataset_path=${dataset2}
com
echo 'done'
#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -J map_wiki
#SBATCH -p cpu
#SBATCH -w node3
#SBATCH --mem=200G
#SBATCH --time=0-06:00:00
#SBATCH --cpus-per-task=16

source /home/pgrimal/.bashrc
source activate cvlp

dataset1=/scratch/pgrimal/wikimage_split
dataset2=/scratch/pgrimal/multimedia_split
wikimage=/scratch/pgrimal/wikimage_no_filter/
title2index=/scratch/pgrimal/title2index.json
key_b_FN=fastrcnn_boxes
key_f_F=fastrcnn_features
key_CLIP=clip_features

echo 'map embed'
echo 'wiki embed'
python -m process_data.map_embedding \
    --dataset_path=${dataset1} \
    --key_features_FasterRCNN=${key_f_F} \
    --key_boxes_FasterRCNN=${key_b_FN} \
    --key_featuresCLIP=${key_CLIP} \
    --wikimage_path=${wikimage} \
    --title2index_wikimage=${title2index}

<<com
echo 'wiki rmv'
python -m process_data.remove_len2 \
    --dataset_path=${dataset1}

echo 'mul rmv'
python -m process_data.remove_len2 \
    --dataset_path=${dataset2}
com
echo 'done'
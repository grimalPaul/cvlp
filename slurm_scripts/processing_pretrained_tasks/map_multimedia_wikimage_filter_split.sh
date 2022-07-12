#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -J analyse
#SBATCH --gres=gpu:1
#SBATCH -p gpu
#SBATCH --mem=40G
#SBATCH --time=0-06:00:00

source /home/pgrimal/.bashrc
source activate cvlp

dataset1=/scratch_global/stage_pgrimal/data/CVLP/data/datasets/wikimage/wikimage_split
dataset2=/scratch_global/stage_pgrimal/data/CVLP/data/datasets/multimedia/filtered/multimedia_split
wikimage=/scratch_global/stage_pgrimal/data/CVLP/data/datasets/wikimage/wikimage_no_filter/
title2index=/scratch_global/stage_pgrimal/data/CVLP/data/datasets/wikimage/title2index.json
key_b_FN=fastrcnn_boxes
key_f_F=fastrcnn_features
key_CLIP=clip_features

python -m process_data.map_embedding \
    --dataset_path=${dataset1}
    --key_features_FasterRCNN=${key_f_FN}
    --key_boxes_FasterRCNN=${key_b_FN}
    --key_featuresCLIP=${key_CLIP}
    --wikimage_path=${wikimage}
    --title2index_wikimage=${title2index}

python -m process_data.map_embedding \
    --dataset_path=${dataset2}
    --key_features_FasterRCNN=${key_f_FN}
    --key_boxes_FasterRCNN=${key_b_FN}
    --key_featuresCLIP=${key_CLIP}
    --wikimage_path=${wikimage}
    --title2index_wikimage=${title2index}

python -m process_data.remove_len2 \
    --dataset_path=${dataset1}

python -m process_data.remove_len2 \
    --dataset_path=${dataset2}
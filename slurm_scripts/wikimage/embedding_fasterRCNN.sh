#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -J embedding_image
#SBATCH --gres=gpu:1
#SBATCH -p gpu
#SBATCH -w node7
#SBATCH --mem=40G
#SBATCH --time=0-06:00:00

source /home/pgrimal/.bashrc
source activate cvlp

python -m processing.embedding_image \
    --dataset_path=/scratch_global/stage_pgrimal/data/CVLP/data/datasets/wikimage/wikimage_no_filter \
    --type=FasterRCNN \
    --model_config_path=/scratch_global/stage_pgrimal/data/CVLP/data/frcnn_model/config.yaml \
    --model=/scratch_global/stage_pgrimal/data/CVLP/data/frcnn_model/pytorch_model.bin \
    --image_path=/scratch_global/stage_pgrimal/data/CVLP/data/datasets/wikimage/Commons_wikimage \
    --key_image=list_images \
    --key_image_embedding=fastrcnn \
    --batch_size=4 \
    --log_path=/home/pgrimal/CVLEP/error_embedding.json

echo "done"

error 
Oscar II of Sweden
Ragnar Hyltén-Cavallius
Trinity College, Oxford
SCARA
GCompris
Vegetable Lamb of Tartary
['Oscar II of Sweden', 'Ragnar Hyltén-Cavallius', 'Trinity College, Oxford', 'SCARA', 'GCompris', 'Vegetable Lamb of Tartary']

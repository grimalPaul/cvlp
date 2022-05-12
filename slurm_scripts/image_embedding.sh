#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -J embedding
#SBATCH --gres=gpu:1
#SBATCH -p gpu
#SBATCH --mem=24G
#SBATCH --time=0-06:00:00

source /home/pgrimal/.bashrc
source activate cvlp

python -m ir.encode_dataset \
    --dataset_path=/scratch_global/stage_pgrimal/data/miniViQuAE/data/wikidata_id/vlt5/vlt5_viquae_dataset \
    --type=embedding_image \
    --model_config_path=/scratch_global/stage_pgrimal/data/CVLP/data/frcnn_model/config.yaml \
    --model=/scratch_global/stage_pgrimal/data/CVLP/data/frcnn_model/pytorch_model.bin \
    --image_path=/scratch_global/stage_pgrimal/data/miniViQuAE/data/dataset/miniCommons/ \
    --key_image=image \
    --key_image_embedding=vlt5

echo "done"

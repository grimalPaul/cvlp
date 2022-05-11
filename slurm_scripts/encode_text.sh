#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -J embedding
#SBATCH --gres=gpu:1
#SBATCH -p gpu
#SBATCH --mem=40G
#SBATCH --time=0-06:00:00

source /home/pgrimal/.bashrc
source activate cvlp

python -m ir.encode_dataset \
    --path_dataset=/scratch_global/stage_pgrimal/data/miniViQuAE/data/wikidata_id/vlt5/vlt5_viquae_dataset \
    --type=encode_text \
    --path_config_model=experiments/model_cvlep/bergamote/encodersT5.json \
    --key_text=input \
    --key_token=vlt5_input_ids \
    --which_tokenizer=question


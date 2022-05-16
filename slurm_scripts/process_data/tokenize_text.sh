#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -J tokenize
#SBATCH --gres=gpu:1
#SBATCH -p gpu
#SBATCH --mem=30G
#SBATCH --time=0-06:00:00

source /home/pgrimal/.bashrc
source activate cvlp

python -m ir.encode_dataset \
    --dataset_path=/scratch_global/stage_pgrimal/data/CVLP/data/datasets/passages \
    --type=tokenize_text \
    --model_config_path=experiments/model_cvlep/bergamote/encodersT5.json \
    --key_text=passage \
    --key_token=vlt5_input_ids \
    --which_tokenizer=passage

echo "done"

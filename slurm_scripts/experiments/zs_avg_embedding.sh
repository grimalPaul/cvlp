#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -J embedding_passage
#SBATCH --gres=gpu:1
#SBATCH -p gpu
#SBATCH -w node4
#SBATCH --mem=30G
#SBATCH --time=7-00:00:00

source /home/pgrimal/.bashrc
source activate cvlp

echo "------------vlt5----------------"
echo "----------vqa-----------"

echo "dataset avg"
python -m processing.embedding_dataset \
    --dataset_path=/scratch_global/stage_pgrimal/data/CVLP/data/datasets/vlt5_viquae_dataset \
    --type=question \
    --model_config_path=experiments/model_cvlep/bergamote/encodersT5.json \
    --key_boxes=vlt5_normalized_boxes \
    --key_vision_features=vlt5_features \
    --key_text=input_vqa \
    --key_embedding=vlt5_embedding_vqa_avg \
    --kb_path=/scratch_global/stage_pgrimal/data/CVLP/data/datasets/kb \
    --batch_size=64 \
    --pool_strategy=avg

echo "passages avg"
python -m processing.embedding_dataset \
    --dataset_path=/scratch_global/stage_pgrimal/data/CVLP/data/datasets/passages \
    --type=passage \
    --model_config_path=experiments/model_cvlep/bergamote/encodersT5.json \
    --key_boxes=vlt5_normalized_boxes \
    --key_vision_features=vlt5_features \
    --key_text=passage_vqa \
    --key_embedding=vlt5_embedding_vqa_avg \
    --kb_path=/scratch_global/stage_pgrimal/data/CVLP/data/datasets/kb \
    --batch_size=64 \
    --pool_strategy=avg


echo "---imt---"
echo "dataset avg"
python -m processing.embedding_dataset \
    --dataset_path=/scratch_global/stage_pgrimal/data/CVLP/data/datasets/vlt5_viquae_dataset \
    --type=question \
    --model_config_path=experiments/model_cvlep/bergamote/encodersT5.json \
    --key_boxes=vlt5_normalized_boxes \
    --key_vision_features=vlt5_features \
    --key_text=input_imt \
    --key_embedding=vlt5_embedding_imt_avg \
    --kb_path=/scratch_global/stage_pgrimal/data/CVLP/data/datasets/kb \
    --batch_size=64 \
    --pool_strategy=avg

echo "passages avg"
python -m processing.embedding_dataset \
    --dataset_path=/scratch_global/stage_pgrimal/data/CVLP/data/datasets/passages \
    --type=passage \
    --model_config_path=experiments/model_cvlep/bergamote/encodersT5.json \
    --key_boxes=vlt5_normalized_boxes \
    --key_vision_features=vlt5_features \
    --key_text=passage_imt \
    --key_embedding=vlt5_embedding_imt_avg \
    --kb_path=/scratch_global/stage_pgrimal/data/CVLP/data/datasets/kb \
    --batch_size=64 \
    --pool_strategy=avg

echo "----------------vlbart------------"
echo "----------vqa-----------"

echo "dataset avg"
python -m processing.embedding_dataset \
    --dataset_path=/scratch_global/stage_pgrimal/data/CVLP/data/datasets/vlbart_viquae_dataset \
    --type=question \
    --model_config_path=experiments/model_cvlep/bergamote/encodersBart.json \
    --key_boxes=vlt5_normalized_boxes \
    --key_vision_features=vlt5_features \
    --key_text=input_vqa \
    --key_embedding=vlbart_embedding_vqa_avg \
    --kb_path=/scratch_global/stage_pgrimal/data/CVLP/data/datasets/kb \
    --batch_size=64 \
    --pool_strategy=avg

echo "passages avg"
python -m processing.embedding_dataset \
    --dataset_path=/scratch_global/stage_pgrimal/data/CVLP/data/datasets/bart_passages \
    --type=passage \
    --model_config_path=experiments/model_cvlep/bergamote/encodersBart.json \
    --key_boxes=vlt5_normalized_boxes \
    --key_vision_features=vlt5_features \
    --key_text=passage_vqa \
    --key_embedding=vlbart_embedding_vqa_avg \
    --kb_path=/scratch_global/stage_pgrimal/data/CVLP/data/datasets/kb \
    --batch_size=64 \
    --pool_strategy=avg

echo "---imt---"
echo "dataset avg"
python -m processing.embedding_dataset \
    --dataset_path=/scratch_global/stage_pgrimal/data/CVLP/data/datasets/vlbart_viquae_dataset \
    --type=question \
    --model_config_path=experiments/model_cvlep/bergamote/encodersBart.json \
    --key_boxes=vlt5_normalized_boxes \
    --key_vision_features=vlt5_features \
    --key_text=input_imt \
    --key_embedding=vlbart_embedding_imt_avg \
    --kb_path=/scratch_global/stage_pgrimal/data/CVLP/data/datasets/kb \
    --batch_size=64 \
    --pool_strategy=avg

echo "passages avg"
python -m processing.embedding_dataset \
    --dataset_path=/scratch_global/stage_pgrimal/data/CVLP/data/datasets/bart_passages \
    --type=passage \
    --model_config_path=experiments/model_cvlep/bergamote/encodersBart.json \
    --key_boxes=vlt5_normalized_boxes \
    --key_vision_features=vlt5_features \
    --key_text=passage_imt \
    --key_embedding=vlbart_embedding_imt_avg \
    --kb_path=/scratch_global/stage_pgrimal/data/CVLP/data/datasets/kb \
    --batch_size=64 \
    --pool_strategy=avg

echo "DONE"

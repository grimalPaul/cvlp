#!/bin/bash
#SBATCH --time=7-00:00:00
#SBATCH --nodes=3
#SBATCH -J resnet_train
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH --mem=50G
#SBATCH --nodelist=node7,node4,node6

source /home/pgrimal/.bashrc
source activate cvlp

export MASTER_ADDR="$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n1)"
export MASTER_PORT=29700

encoder_question_path=experiments/config_vladapter/multitasks/resnet_simple_adapter/encoder_simple_adapter.json
encoder_passage_path=experiments/config_vladapter/multitasks/resnet_simple_adapter/encoder_simple_adapter.json
model_path=experiments/config_vladapter/multitasks/resnet_simple_adapter/config_model.json
training_path=experiments/config_vladapter/multitasks/resnet_simple_adapter/training_multitask.json

echo "+ projection et adapter"
srun --kill-on-bad-exit=1 python -m cvlep.trainer_multitask \
    --encoder_question_path=${encoder_question_path} \
    --encoder_passage_path=${encoder_passage_path} \
    --model_path=${model_path} \
    --training_path=${training_path}

echo "The End"
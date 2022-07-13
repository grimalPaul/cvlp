#!/bin/bash
#SBATCH --time=7-00:00:00
#SBATCH --nodes=1
#SBATCH -J test_training
#SBATCH --gres=gpu:2
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=5
#SBATCH --mem=46G
#SBATCH --nodelist=node7

source /home/pgrimal/.bashrc
source activate cvlp

export MASTER_ADDR="$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n1)"
export MASTER_PORT=29700

encoder_question_path=experiments/config_vladapter/multitasks/test/encoder_simple_adapter.json
encoder_passage_path=experiments/config_vladapter/bergamote/adapter_projection/encoder_simple_adapter.json
model_path=experiments/config_vladapter/multitasks/test/config_model.json
training_path=experiments/config_vladapter/multitasks/test/training_multitask.json

echo "+ projection et adapter"
srun --kill-on-bad-exit=1 python -m cvlep.trainer_multitask \
    --encoder_question_path=${encoder_question_path} \
    --encoder_passage_path=${encoder_passage_path} \
    --model_path=${model_path} \
    --training_path=${training_path}

echo "The End"
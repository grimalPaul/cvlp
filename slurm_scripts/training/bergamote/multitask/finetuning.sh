#!/bin/bash
#SBATCH --time=7-00:00:00
#SBATCH --nodes=2
#SBATCH -J fine_tuning_bs48
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=4
#SBATCH --mem=46G
#SBATCH --nodelist=node7,node6

source /home/pgrimal/.bashrc
source activate cvlp

export MASTER_ADDR="$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n1)"
export MASTER_PORT=29700

encoder_question_path=experiments/config_vladapter/bergamote/finetuning/encoder_question.json
encoder_passage_path=experiments/config_vladapter/bergamote/finetuning/encoder_passage.json
model_path=experiments/config_vladapter/bergamote/finetuning/config_model.json
training_path=experiments/config_vladapter/bergamote/finetuning/training_finetuning.json

echo "finetuning bs 64"
srun --kill-on-bad-exit=1 python -m cvlep.trainer_base_vladapter \
    --encoder_question_path=${encoder_question_path} \
    --encoder_passage_path=${encoder_passage_path} \
    --model_path=${model_path} \
    --training_path=${training_path}

echo "The End"


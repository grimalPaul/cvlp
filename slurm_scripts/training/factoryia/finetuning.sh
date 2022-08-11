#!/usr/bin/env sh
#SBATCH --time=2-00:00:00
#SBATCH --nodes=1
#SBATCH -J finetuning
#SBATCH --gres=gpu:2
#SBATCH --partition=gpuv100
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=5
#SBATCH -w node27
#SBATCH --mem=60G

source /home/users/pgrimal/.bashrc
source activate cvlp

echo "finetuning"

export MASTER_ADDR="$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n1)"
export MASTER_PORT=29705

encoder_question_path=experiments/config_vladapter/factoryIA/finetuning/encoder_question.json
encoder_passage_path=experiments/config_vladapter/factoryIA/finetuning/encoder_passage.json
model_path=experiments/config_vladapter/factoryIA/finetuning/config_model.json
training_path=experiments/config_vladapter/factoryIA/finetuning/finetuning.json

echo "fine tuning"
srun --kill-on-bad-exit=1 python -m cvlep.trainer_base_vladapter \
    --encoder_question_path=${encoder_question_path} \
    --encoder_passage_path=${encoder_passage_path} \
    --model_path=${model_path} \
    --training_path=${training_path}

echo "The End"


#!/bin/bash
#SBATCH --time=0-02:00:00
#SBATCH -w node7
#SBATCH -J prompt
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=5
#SBATCH --mem=46G

source /home/pgrimal/.bashrc
source activate cvlp

echo "CUDA devices : $CUDA_VISIBLE_DEVICES"

export MASTER_ADDR="$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n1)"
export MASTER_PORT=29700

encoder_question_path=experiments/config_vladapter/bergamote/prompt/encoder_prompting.json
encoder_passage_path=experiments/config_vladapter/bergamote/prompt/encoder_prompting.json
model_path=experiments/config_vladapter/bergamote/prompt/config_model.json
training_path=experiments/config_vladapter/bergamote/prompt/training_prompt.json

echo "Prompt tuning"
srun --kill-on-bad-exit=1 python -m cvlep.trainer_base_vladapter \
    --encoder_question_path=${encoder_question_path} \
    --encoder_passage_path=${encoder_passage_path} \
    --model_path=${model_path} \
    --training_path=${training_path}

echo "The End"
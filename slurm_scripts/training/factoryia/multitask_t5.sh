#!/usr/bin/env sh
#SBATCH --time=7-00:00:00
#SBATCH --nodes=1
#SBATCH -J train_clipT5
#SBATCH --gres=gpu:2
#SBATCH --partition=gpuv100
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=7
#SBATCH -w node27
#SBATCH --mem=45G

source /home/users/pgrimal/.bashrc
source activate cvlp2

export MASTER_ADDR="$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n1)"
export MASTER_PORT=29702

encoder_question_path=experiments/config_vladapter/factoryIA/T5/clip_simple_adapter_v100/encoder_simple_adapter.json
encoder_passage_path=experiments/config_vladapter/factoryIA/T5/clip_simple_adapter_v100/encoder_simple_adapter.json
model_path=experiments/config_vladapter/factoryIA/T5/clip_simple_adapter_v100/config_model.json
training_path=experiments/config_vladapter/factoryIA/T5/clip_simple_adapter_v100/training_multitask.json

echo "Training model with clip embedding"
srun --kill-on-bad-exit=1 python -m cvlep.trainer_multitask \
    --encoder_question_path=${encoder_question_path} \
    --encoder_passage_path=${encoder_passage_path} \
    --model_path=${model_path} \
    --training_path=${training_path}

echo "The End"
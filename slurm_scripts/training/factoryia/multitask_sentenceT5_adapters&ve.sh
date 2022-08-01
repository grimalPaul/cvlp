#!/usr/bin/env sh
#SBATCH --time=7-00:00:00
#SBATCH --nodes=2
#SBATCH -J sT5_all
#SBATCH --gres=gpu:4
#SBATCH --partition=classicgpu,gpup100,gpuv100
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=7
#SBATCH --mem=45G

source /home/users/pgrimal/.bashrc
source activate cvlp2

export MASTER_ADDR="$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n1)"
export MASTER_PORT=29701

encoder_question_path=experiments/config_vladapter/factoryIA/sentenceT5/clip_learn_adapter_&_ve/encoder_question.json
encoder_passage_path=experiments/config_vladapter/factoryIA/sentenceT5/clip_learn_adapter_&_ve/encoder_passage.json
model_path=experiments/config_vladapter/factoryIA/sentenceT5/clip_learn_adapter_&_ve/config_model.json
training_path=experiments/config_vladapter/factoryIA/sentenceT5/clip_learn_adapter_&_ve/training_multitask.json

echo "adapter & ve"
srun --kill-on-bad-exit=1 python -m cvlep.trainer_multitask \
    --encoder_question_path=${encoder_question_path} \
    --encoder_passage_path=${encoder_passage_path} \
    --model_path=${model_path} \
    --training_path=${training_path}

echo "The End"
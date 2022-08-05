#!/usr/bin/env sh
#SBATCH --time=7-00:00:00
#SBATCH --nodes=1
#SBATCH -J sT5_v100
#SBATCH --gres=gpu:4
#SBATCH --partition=gpuv100
#SBATCH -w node27
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --mem=45G

source /home/users/pgrimal/.bashrc
source activate cvlp2

export MASTER_ADDR="$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n1)"
export MASTER_PORT=29709

encoder_question_path=experiments/config_vladapter/factoryIA/sentenceT5/st5_clip_v100/encoder_question.json
encoder_passage_path=experiments/config_vladapter/factoryIA/sentenceT5/st5_clip_v100/encoder_passage.json
model_path=experiments/config_vladapter/factoryIA/sentenceT5/st5_clip_v100/config_model.json
training_path=experiments/config_vladapter/factoryIA/sentenceT5/st5_clip_v100/training_multitask.json

echo "big batch size"
echo "save snap/sT5_adapter_ve_v100/",
echo "tensorboard tensorboard/st5_adapter_v100/"


srun --kill-on-bad-exit=1 python -m cvlep.trainer_multitask \
    --encoder_question_path=${encoder_question_path} \
    --encoder_passage_path=${encoder_passage_path} \
    --model_path=${model_path} \
    --training_path=${training_path}

echo "The End"
#!/usr/bin/env sh
#SBATCH --time=7-00:00:00
#SBATCH --nodes=1
#SBATCH -J train_clipT5
#SBATCH --gres=gpu:4
#SBATCH --partition=classicgpu,gpup100,gpuv100
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=7
#SBATCH --mem=40G


source /home/users/pgrimal/.bashrc
source activate cvlp2

export MASTER_ADDR="$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n1)"
export MASTER_PORT=29704

encoder_question_path=experiments/config_vladapter/factoryIA/T5/clip_visu_embedencoder_question.json
encoder_passage_path=experiments/config_vladapter/factoryIA/T5/clip_visu_embedencoder_passage.json
model_path=experiments/config_vladapter/factoryIA/T5/clip_visu_embedconfig_model.json
training_path=experiments/config_vladapter/factoryIA/T5/clip_visu_embedtraining_multitask.json

echo "clip_sentenceT5_vepretrained"
srun --kill-on-bad-exit=1 python -m cvlep.trainer_multitask \
    --encoder_question_path=${encoder_question_path} \
    --encoder_passage_path=${encoder_passage_path} \
    --model_path=${model_path} \
    --training_path=${training_path}

echo "The End"
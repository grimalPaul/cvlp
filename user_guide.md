# User Guide

## Training

### FactoryIA

Create a configuration like in `experiments/config_vladapter/factoryIA/sentenceT5` or in `experiments/config_vladapter/factoryIA/T5`

and run the slurm script.

- factoryIA : my script for factoryIA are here : `slurm_scripts/training/factoryia`

```bash
#!/usr/bin/env sh
#SBATCH --time=7-00:00:00
#SBATCH --nodes=3
#SBATCH -J train_clipT5
#SBATCH --gres=gpu:4
#SBATCH --partition=classicgpu
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=7
#SBATCH --mem=40G

source /home/users/pgrimal/.bashrc
source activate cvlp2

export MASTER_ADDR="$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n1)"
export MASTER_PORT=29701

encoder_question_path=experiments/config_vladapter/factoryIA/T5/clip_simple_adapter/encoder_simple_adapter.json
encoder_passage_path=experiments/config_vladapter/factoryIA/T5/clip_simple_adapter/encoder_simple_adapter.json
model_path=experiments/config_vladapter/factoryIA/T5/clip_simple_adapter/config_model.json
training_path=experiments/config_vladapter/factoryIA/T5/clip_simple_adapter/training_multitask.json

echo "Training model with clip embedding"
srun --kill-on-bad-exit=1 python -m cvlep.trainer_multitask \
    --encoder_question_path=${encoder_question_path} \
    --encoder_passage_path=${encoder_passage_path} \
    --model_path=${model_path} \
    --training_path=${training_path}

echo "The End"
```

- bagermote : my script are here : `slurm_scripts/training/bergamote/multitask`

```bash
#!/bin/bash
#SBATCH --time=7-00:00:00
#SBATCH --nodes=2
#SBATCH -J resnet_train
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
##SBATCH --nodelist=node7,node4,node6
#SBATCH --nodelist=node4,node6

source /home/pgrimal/.bashrc
source activate cvlp

export MASTER_ADDR="$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n1)"
export MASTER_PORT=29701

encoder_question_path=experiments/config_vladapter/bergamote/T5/resnet_simple_adapter/encoder_simple_adapter.json
encoder_passage_path=experiments/config_vladapter/bergamote/T5/resnet_simple_adapter/encoder_simple_adapter.json
model_path=experiments/config_vladapter/bergamote/T5/resnet_simple_adapter/config_model.json
training_path=experiments/config_vladapter/bergamote/T5/resnet_simple_adapter/training_multitask.json

echo "faster r cnn T5"
srun --kill-on-bad-exit=1 python -m cvlep.trainer_multitask \
    --encoder_question_path=${encoder_question_path} \
    --encoder_passage_path=${encoder_passage_path} \
    --model_path=${model_path} \
    --training_path=${training_path}

echo "The End"
```

## Finetuning

You have to duplicate the encoder config in two configuration because after pretraining encoder question and encoder passage are different. Add the path where is save the model next to `load_path`.

The config for the training is different than in pretrained step. We do not have multiple datasets.

- factoryIA : my script for factoryIA are here : `slurm_scripts/training/factoryia`
- bagermote : my script are here : `slurm_scripts/training/bergamote/multitask`

## Embedding

## Research

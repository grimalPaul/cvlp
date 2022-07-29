# User Guide

## PreTraining

Create a configuration like in `experiments/config_vladapter/factoryIA/sentenceT5` or in `experiments/config_vladapter/factoryIA/T5`

and run the slurm script.

- factoryIA : my script for factoryIA are here : `slurm_scripts/training/factoryia` and config are `experiments/config_vladapter/factoryIA/T5` and `experiments/config_vladapter/factoryIA/sentenceT5`.

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

- bagermote : my script are here : `slurm_scripts/training/bergamote/multitask` and config are `experiments/config_vladapter/bergamote/T5/resnet_simple_adapter` and `experiments/config_vladapter/bergamote/sentenceT5`

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

You have to duplicate the encoder config in two configuration because after pretraining encoder question and encoder passage are different. Add the path where the model is save next to `load_path`.

The config for the training is different than in pretrained step. We do not have multiple datasets.

- factoryIA : my script for factoryIA are here : `slurm_scripts/training/factoryia`
- bagermote : my script are here : `slurm_scripts/training/bergamote/multitask/finetuning`

## Embedding

Embed dataset and knowledge base
instead of vision_features and vision_boxes with faster RCNN write

```bash
--key_boxes=fastrcnn_boxes \
--key_vision_features=fastrcnn_features \
```

And for clip write :

```bash
--key_vision_features=clip_features \
# You must not write --key_boxes=... because we dont use boxes with clip

```

```bash
echo "embedding passage"
python -m processing.embedding_dataset \
    --dataset_path=${passages} \
    --type=passage \
    --config_question_path=${config_question_path} \
    --config_passage_path=${config_passage_path} \
    --config_model_path=${config_model_path} \
    --key_boxes=vision_boxes \
    --key_vision_features=vision_features \
    --key_text=passage \
    --key_embedding=${key_in_passage} \
    --kb_path=${kb} \
    --batch_size=${batch_size}

echo "embedding dataset"

python -m processing.embedding_dataset \
    --dataset_path=${dataset} \
    --type=question \
    --config_question_path=${config_question_path} \
    --config_passage_path=${config_passage_path} \
    --config_model_path=${config_model_path} \
    --key_boxes=vision_boxes \
    --key_vision_features=vision_features \
    --key_text=input \
    --key_embedding=${key_in_dataset} \
    --batch_size=${batch_size}
```

And then compute the different metrics :

```bash
python -m search \
    --dataset_path=viquae/test \
    --config=experiments/ir/VL/experiments/clip_multitask/multitask.json \
    --metrics_path=experiments/ir/VL/clip_multitask/ \
    --k=100 \
    --batch_size=64
```

## Config

```json
{
    "backbone": "path to the backbone",
    "init_model_path":"t5_pretrained or sentence T5",
    "tokenizer": "path to tokenizer",
    "feat_dim": 2048,
    "pos_dim": 4,
    
    "add_projectionHead":false,#add or not projection head at the output of the encoder
    "dim_projectionHead":768,
    
    # dont change
    "use_vision": true,
    "use_vis_order_embedding": true,
    "additional_visual_embedding_layers": 0,
    "use_vis_layer_norm": true,
    "individual_vis_layer_norm": true,
    "share_vis_lang_layer_norm": false,
    "use_lm_head_adapter": false,
    "n_boxes": 36,
    "max_n_boxes": 36,
    "expand_vis_embedding": false,
    "n_image_tokens": null,
    "vis_use_transformer": false,
    "vis_pooling_output": false,
    "sparse_sample": false,
    "oneddownsample": false,
    
    "downsample": false, # true with clip, false with faster rcnn
    "use_adapter": true, # true if you want use adapter
    "use_hyperformer": false,
    "use_compacter": false,
    "use_lradapter": false,
    "add_adapter_cross_attn": false,
    "tasks": "IR",
    "use_single_prompt": false,
    "encoder_prompt_len": 0,
    "mid_dim": 0,
    "decoder_prompt_len": 0,
    "reduction_factor": 8, # reduction factor of the adapter
    "use_lora": false,
    "do_lower_case": false,
    "dropout": 0.1,
    "load_path":"path to load model from other training",
    "from_scratch": false,
    
    # unfreeze or not some part of the model
    "unfreeze_visual_embedding": true,
    "unfreeze_language_model": false,
    "unfreeze_lm_head": false,
    "unfreeze_vis_encoder": false,
    "unfreeze_vis_last_layer": false,
    "use_vis_adapter": false,
    "unfreeze_layer_norms": true,
    "unfreeze_batch_norms": false,
    "unfreeze_projectionHead":false,

    # dont touch
    "unique_hyper_net": false,
    "efficient_unique_hyper_net": false,
    "use_single_adapter": false,
    "hypercomplex_division": false,
    "phm_rank": false,
    "shared_phm_rule": false,
    "factorized_phm": false,
    "low_rank_rank": false,
    "phm_init_range": false,
    "share_down_sampler": false,
    "share_up_sampler": false,
    "shared_phm_rule_over_tasks": false,
    "add_layer_norm_before_adapter": false,
    "add_layer_norm_after_adapter": false,
    "track_z": false,
    "projected_task_embedding_dim": -1
}
```

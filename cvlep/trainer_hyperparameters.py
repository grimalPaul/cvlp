# optimize scheduler, optimize learning rate, optimize reducto factor

# je pense que je vais devoir charger les configs en dehors du trainer
# cela permettra facilement de modifier le modele

import argparse
import optuna
import os
from cvlep.VLT5.param import Config


# semble possible avec cela

https://github.com/optuna/optuna-examples/blob/main/pytorch/pytorch_distributed_simple.py

len prompt encoder
mid dim prompt encoder

# prompt 
mid_dim
trial = optuna.Trial
trial.suggest_int
trial.suggest_float
trial.suggest
# adapters
reduction factors


objective_fct
study = optuna.cvreate_study(direction='maximize')
study.optimize(objective_fct, n_trials=100)


def objective(trial: optuna.trial.Trial) -> float:
    
    # faire config de training
    config_encoder_question.ATTRIBUT = trial.suggest
    
    # config_encoder_question
    trial.suggest_categorical()

    # config_encoder_passage

    # build dataloader

    # build trainer
    trainer = None

    # return loss
    return trainer.train()

def train(trial):

    trial.report(accuracy, epoch_num)

    if trial.should_prune():

        raise optuna.exceptions.TrialPruned()

    return accuracy

study = optuna.create_study(direction="maximize", sampler=optuna.samplers., pruner=optuna.pruners)

def main_worker(config_training, args):

    config_encoder_question = Config.load_json(args.encoder_question_path)
    config_encoder_passage = Config.load_json(args.encoder_passage_path)
    config_model = Config.load_json(args.model_path)

    





if __name__ == '__main__':
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:  # torchrun launch
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
    elif int(os.environ.get('SLURM_NPROCS', 1)) > 1:  # slurm launch
        rank = int(os.environ["SLURM_PROCID"])
        local_rank = int(os.environ["SLURM_LOCALID"])
        world_size = int(os.environ["SLURM_NPROCS"])
    else:  # single gpu & process launch
        rank = 0
        local_rank = 0
        world_size = 0

    parser = argparse.ArgumentParser()
    parser.add_argument('--encoder_question_path', type=str, required=True)
    parser.add_argument('--encoder_passage_path', type=str, required=True)
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--training_path', type=str, required=True)
    args = parser.parse_args()

    # Training config
    config_training = Config.load_json(args.training_path)
    config_training.world_size = world_size
    config_training.rank = rank
    config_training.local_rank = local_rank
    if world_size > 1:
        config_training.distributed = True
        config_training.multiGPU = True
    else:
        config_training.distributed = False
        config_training.multiGPU = False

    main_worker(config_training, args)
"""
    "unfreeze_visual_embedding":false,
    "unfreeze_language_model":false,
    "unfreeze_lm_head":false,
    "unfreeze_vis_encoder":false,
    "unfreeze_vis_last_layer":false,
    "use_vis_adapter":false,
    "unfreeze_layer_norms":false,
    "unfreeze_batch_norms":false
"""
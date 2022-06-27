# optimize scheduler, optimize learning rate, optimize reducto factor

# je pense que je vais devoir charger les configs en dehors du trainer
# cela permettra facilement de modifier le modele

import argparse
from unicodedata import name
import optuna
import os

from zmq import device
from cvlep.VLT5.param import Config
from cvlep.trainer_base_vladapter import Trainer
from torch import distributed as dist
import torch
from cvlep.viquae_data import get_loader
from packaging import version
from tqdm import tqdm
from cvlep.VLT5.utils import LossMeter
from optuna.trial import TrialState

# Check if Pytorch version >= 1.6 to switch between Native AMP and Apex
if version.parse(torch.__version__) < version.parse("1.6"):
    raise NotImplementedError("We did not implement apex")
else:
    _use_native_amp = True
    from torch.cuda.amp import autocast

class Objective_adapter(object):
    def __init__(self, config_training, config_encoder_passage, config_encoder_question, config_model):
        self.config_training = config_training
        self.config_encoder_passage = config_encoder_passage
        self.config_encoder_question = config_encoder_question
        self.config_model = config_model

    def __call__(self, single_trial):
        trial = optuna.integration.TorchDistributedTrial(trial = single_trial, device=torch.device(self.config_training.local_rank))

        # parameters to optimize
        self.config_encoder_passage.reduction_factor = trial.suggest_int(
            "reduction_factor", 1, 2)

        verbose = True
        if self.config_training.distributed:
            if self.config_training.rank != 0:
                verbose = False
        if verbose:
            print(f"World size : {self.config_training.world_size}")

        if self.config_training.train:
            train_loader = get_loader(
                cls="dpr",
                mode='train',
                batch_size=self.config_training.batch_size,
                seed=self.config_training.seed,
                distributed=self.config_training.distributed,
                workers=self.config_training.num_workers,
                tokenizer_path=self.config_training.tokenizer_path,
                dataset_path=self.config_training.dataset_path,
                kb_path=self.config_training.kb_path,
                passages_path=self.config_training.passages_path,
                key_relevant=self.config_training.key_relevant,
                key_text_question=self.config_training.key_text_question,
                key_text_passage=self.config_training.key_text_passage,
                key_vision_features=self.config_training.key_vision_features,
                key_vision_boxes=self.config_training.key_vision_boxes,
                split='train',
                key_irrelevant=self.config_training.key_irrelevant,
                verbose=verbose
            )
            val_loader = get_loader(
                cls="dpr",
                mode='eval',
                batch_size=self.config_training.valid_batch_size,
                seed=self.config_training.seed,
                distributed=self.config_training.distributed,
                workers=self.config_training.num_workers,
                tokenizer_path=self.config_training.tokenizer_path,
                dataset_path=self.config_training.dataset_path,
                kb_path=self.config_training.kb_path,
                passages_path=self.config_training.passages_path,
                key_relevant=self.config_training.key_relevant,
                key_text_question=self.config_training.key_text_question,
                key_text_passage=self.config_training.key_text_passage,
                key_vision_features=self.config_training.key_vision_features,
                key_vision_boxes=self.config_training.key_vision_boxes,
                split='validation',
                key_irrelevant=self.config_training.key_irrelevant,
                verbose=verbose
            )

            test_loader = None
        elif self.config_training.test:
            test_loader = get_loader(
                cls="dpr",
                mode='test',
                batch_size=self.config_training.test_batch_size,
                seed=self.config_training.seed,
                distributed=self.config_training.distributed,
                workers=self.config_training.num_workers,
                tokenizer_path=self.config_training.tokenizer_path,
                dataset_path=self.config_training.dataset_path,
                kb_path=self.config_training.kb_path,
                passages_path=self.config_training.passages_path,
                key_relevant=self.config_training.key_relevant,
                key_text_question=self.config_training.key_text_question,
                key_text_passage=self.config_training.key_text_passage,
                key_vision_features=self.config_training.key_vision_features,
                key_vision_boxes=self.config_training.key_vision_boxes,
                split='test',
                key_irrelevant=self.config_training.key_irrelevant,
                verbose=verbose
            )
            train_loader = None
            val_loader = None

        trainer = optimizeTrainer(
            config_question=self.config_encoder_question,
            config_passage=self.config_encoder_passage,
            config_model=self.config_model,
            config_training=config_training,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            train=self.config_training.train
        )
        return trainer.train(trial)


class optimizeTrainer(Trainer):
    def __init__(self, config_question, config_passage, config_model, config_training, train_loader=None, val_loader=None, test_loader=None, train=True):
        super().__init__(config_question, config_passage, config_model,
                         config_training, train_loader, val_loader, test_loader, train)
        if train and self.verbose:
            self.writer = "None"

    def train(self, trial):
        if self.verbose:
            best_valid = 0.
            best_epoch = 0
        if self.args.distributed:
            # to always have the same validation loader
            self.val_loader.sampler.set_epoch(0)

        for epoch in tqdm(range(self.args.epochs)):
            if self.verbose:
                loss_meter = LossMeter()
                pbar = tqdm(total=len(self.train_loader), ncols=100)
            self.model.train()
            if self.args.distributed:
                self.train_loader.sampler.set_epoch(epoch)
            for step_i, batch in enumerate(self.train_loader):
                if self.args.fp16 and _use_native_amp:
                    with autocast():
                        if self.args.distributed:
                            loss = self.compute_loss(batch)
                else:
                    loss = self.compute_loss(batch)

                # loss.backward
                if self.args.fp16 and _use_native_amp:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()

                loss = loss.detach()
                # Update Parameters
                if self.args.clip_grad_norm > 0:
                    if self.args.fp16 and _use_native_amp:
                        self.scaler.unscale_(self.optim)
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.args.clip_grad_norm)
                    else:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.args.clip_grad_norm)

                # optim step
                update = True
                if self.args.gradient_accumulation_steps > 1:
                    if step_i == 0:
                        update = False
                    elif step_i % self.args.gradient_accumulation_steps == 0 or step_i == len(self.train_loader) - 1:
                        update = True
                    else:
                        update = False

                if update:
                    if self.args.fp16 and _use_native_amp:
                        self.scaler.step(self.optim)
                        self.scaler.update()
                    else:
                        self.optim.step()

                    if self.lr_scheduler:
                        self.lr_scheduler.step()
                    for param in self.model.parameters():
                        param.grad = None

                # Scheduler
                if self.lr_scheduler:
                    if version.parse(torch.__version__) >= version.parse("1.4"):
                        lr = self.lr_scheduler.get_last_lr()[0]
                    else:
                        lr = self.lr_scheduler.get_lr()[0]
                else:
                    try:
                        lr = self.optim.get_lr()[0]
                    except AttributeError:
                        lr = self.args.lr

                if self.verbose:
                    loss_meter.update(loss.item())
                    desc_str = f'Epoch {epoch} | LR {lr:.6f}'
                    desc_str += f' | Loss {loss_meter.val:4f}'
                    pbar.set_description(desc_str)
                    pbar.update(1)

            if self.args.distributed:
                dist.barrier()

            if self.verbose:
                pbar.close()
                # self.writer.add_scalar('Training_loss', loss_meter.val, epoch)
                # self.writer.add_scalar('lr', lr, epoch)
                # self.writer.flush()
            # Validation
            if self.val_loader is not None:
                self.model.eval()
                with torch.no_grad():
                    if self.verbose:
                        loss_meter = LossMeter()
                        pbar = tqdm(total=len(self.val_loader), ncols=100)
                    for step_i, batch in enumerate(self.val_loader):
                        if self.args.fp16 and _use_native_amp:
                            with autocast():
                                if self.args.distributed:
                                    loss = self.compute_loss(batch)
                        else:
                            loss = self.compute_loss(batch)
                        if self.verbose:
                            loss_meter.update(loss.item())
                            desc_str = f'Validation {epoch} | Loss {loss_meter.val:4f}'
                            pbar.set_description(desc_str)
                            pbar.update(1)
                    if self.verbose:
                        pbar.close()
                        # self.writer.add_scalar(
                        #    'Validation_loss', loss_meter.val, epoch)
                        # self.writer.flush()
                if self.verbose:
                    if loss_meter.val < best_valid or epoch == 0:
                        best_valid = loss_meter.val
                        best_epoch = epoch
                        if self.args.distributed:
                            if self.args.rank == 0:
                                self.save(f"best_{epoch}")
                        else:
                            self.save(f"best_{epoch}")
                    elif epoch % 5 == 0:
                        if self.args.distributed:
                            if self.args.rank == 0:
                                self.save(f"e_{epoch}")
                        else:
                            self.save(f"best_{epoch}")
                    log_str = f"\nEpoch {epoch}/{self.args.epochs - 1}: Valid Loss {loss_meter.val:4f}"
                    log_str += f"\nBest Epoch {best_epoch}: Best Valid Loss {best_valid:4f}"
                    print(log_str)

            if self.args.distributed:
                dist.barrier()
            if self.verbose:
                trial.report(loss_meter.val, epoch)
             
             # Handle pruning based on the intermediate value.
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
        
        if self.verbose:
            # self.writer.close()
            pass
        return loss_meter.val

# semble possible avec cela
# https: // github.com/optuna/optuna-examples/blob/main/pytorch/pytorch_distributed_simple.py

def main_worker(config_training, args):

    config_encoder_question = Config.load_json(args.encoder_question_path)
    config_encoder_passage = Config.load_json(args.encoder_passage_path)
    config_model = Config.load_json(args.model_path)

    print(
            f'Process Launching at GPU {config_training.local_rank} and rank is {config_training.rank}')

    if config_training.distributed and not dist.is_initialized():
        dist.init_process_group(
            backend='nccl', world_size=config_training.world_size, rank=config_training.rank)
    if config_training.distributed:
        torch.device("cuda", index=config_training.local_rank)
    else:
        torch.device("cpu")

    study = None
    n_trials = 20
    if config_training.rank == 0:
        study = optuna.create_study(direction="minimize", study_name="TEST")
        study.optimize(Objective_adapter(config_training, config_encoder_passage, config_encoder_question, config_model), n_trials=n_trials)
    else:
        for _ in range(n_trials):
            try:
                obj = Objective_adapter(config_training, config_encoder_passage, config_encoder_question, config_model)
                obj(None)
            except optuna.TrialPruned:
                pass
    if config_training.rank == 0:
        assert study is not None
        pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
        complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

        print("Study statistics: ")
        print("  Number of finished trials: ", len(study.trials))
        print("  Number of pruned trials: ", len(pruned_trials))
        print("  Number of complete trials: ", len(complete_trials))

        print("Best trial:")
        trial = study.best_trial

        print("  Value: ", trial.value)

        print("  Params: ")
        for key, value in trial.params.items():
            print("    {}: {}".format(key, value))

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

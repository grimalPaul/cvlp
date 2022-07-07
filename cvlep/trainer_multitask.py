import argparse
from cvlep.multitask_data import get_multitask_loader, get_val_loader
from cvlep.pretrain_data import get_loader
from cvlep.trainer_base_vladapter import Trainer
from tqdm import tqdm
from packaging import version
import torch
from cvlep.VLT5.utils import LossMeter
from torch import distributed as dist
from cvlep.VLT5.param import Config
import os
import json

# Check if Pytorch version >= 1.6 to switch between Native AMP and Apex
if version.parse(torch.__version__) < version.parse("1.6"):
    raise NotImplementedError("We did not implement apex")
else:
    _use_native_amp = True
    from torch.cuda.amp import autocast


class Trainer_Multitask(Trainer):
    def __init__(self, config_question, config_passage, config_model, config_training, train_loader=None, val_loader=None, test_loader=None, train=True, local=False):
        super().__init__(config_question, config_passage, config_model,
                         config_training, train_loader, val_loader, test_loader, train, local)

    def train(self):
        if self.verbose:
            best_valid = 0.
            best_epoch = 0
        if self.args.distributed and self.val_loader is not None:
            for task_name in self.val_loader.keys():
                self.val_loader[task_name].set_epoch(0)

        # TODO : init counter for class
        task_counter = {}

        for epoch in tqdm(range(self.args.epochs)):
            if self.verbose:
                tasks_loss = {}
                for task in task_counter.keys():
                    tasks_loss[task] = LossMeter()
                loss_meter = LossMeter()
                pbar = tqdm(total=len(self.train_loader), ncols=100)
            self.model.train()
            if self.args.distributed:
                self.train_loader.set_epoch(epoch)

            for step_i, batch in enumerate(self.train_loader):
                task = batch['task']
                task_counter[task] += 1

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
                    tasks_loss[task].update(loss.item())
                    loss_meter.update(loss.item())
                    desc_str = f'Epoch {epoch} | LR {lr:.6f}'
                    for task_name, nb in task_counter.items():
                        desc_str += f'| {task_name} : {nb} '
                        if tasks_loss[task_name].val > 0:
                            desc_str += f'loss : {tasks_loss[task_name].val:4f}'
                    desc_str += f' | Loss Total {loss_meter.val:4f}'
                    pbar.set_description(desc_str)
                    pbar.update(1)

            if self.args.distributed:
                dist.barrier()

            if self.verbose:
                pbar.close()
                self.writer.add_scalar('Training_loss', loss_meter.val, epoch)
                self.writer.add_scalar('lr', lr, epoch)
                for task_name, l in tasks_loss.items():
                    self.writer.add_scalar(f'{task_name}_loss', l.val, epoch)
                self.writer.flush()

            # Validation
            if self.val_loader is not None:
                self.model.eval()
                # val loader
                # {
                # task:val_loader,
                # task:val_loader,
                # task:val_loader
                # }
                with torch.no_grad():
                    if self.verbose:
                        tasks_loss = {}
                        for task in task_counter.keys():
                            tasks_loss[task] = LossMeter()
                        loss_meter = LossMeter()
                        size = 0
                        for loader in self.val_loader.values():
                            size += len(loader)
                        pbar = tqdm(total=size, ncols=100)
                    for t, loader in self.val_loader:
                        for batch in loader:
                            task = batch['task']
                            if self.args.fp16 and _use_native_amp:
                                with autocast():
                                    if self.args.distributed:
                                        loss = self.compute_loss(batch)
                            else:
                                loss = self.compute_loss(batch)
                            if self.verbose:
                                tasks_loss[task].update(loss.item())
                                loss_meter.update(loss.item())
                                desc_str = f'Validation {epoch} | Loss {loss_meter.val:4f}'
                                for task_name, l in tasks_loss.items():
                                    if l.val > 0:
                                        desc_str += f'| {task_name}_loss : {l.val:4f}'
                                pbar.set_description(desc_str)
                                pbar.update(1)
                    if self.verbose:
                        pbar.close()
                        for task_name, l in tasks_loss.items():
                            self.writer.add_scalar(
                                f'Validation_{task_name}_loss', l.val, epoch)
                        self.writer.add_scalar(
                            'Validation_loss', loss_meter.val, epoch)
                        self.writer.flush()
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
            self.writer.close()

    def compute_loss(self, batch):
        # Calculates In-batch negatives schema loss and supports to run it in DDP mode by exchanging the representations across all the nodes.
        # From https://github.com/PaulLerner/ViQuAE/blob/e032dedc568c8a56b9a54ada6bb4dfa20c4301de/meerqat/train/trainer.py#L206

        if self.args.distributed:
            output_question, output_context, local_labels = self.model.module.train_step(
                batch)
        else:
            output_question, output_context, local_labels = self.model.train_step(
                batch)

        # N question in the batch * dim model = N * d
        local_question_representations = output_question
        # (1 relevant + 1 irrelevant) * N * dim model = 2N * d
        local_context_representations = output_context

        if self.args.world_size > 1:
            question_representations_to_send = torch.empty_like(
                local_question_representations).copy_(local_question_representations).detach_()
            context_representations_to_send = torch.empty_like(
                local_context_representations).copy_(local_context_representations).detach_()
            labels_to_send = torch.empty_like(local_labels).copy_(local_labels)

            # gathers representations from other GPUs
            question_representations_gatherer = [torch.empty_like(
                question_representations_to_send) for _ in range(self.args.world_size)]
            context_representations_gatherer = [torch.empty_like(
                context_representations_to_send) for _ in range(self.args.world_size)]
            labels_gatherer = [torch.empty_like(
                labels_to_send) for _ in range(self.args.world_size)]
            dist.all_gather(question_representations_gatherer,
                            question_representations_to_send)
            dist.all_gather(context_representations_gatherer,
                            context_representations_to_send)
            dist.all_gather(labels_gatherer, labels_to_send)

            # keep local vector in the local_rank index (taken from DPR, to not loose the gradients?)
            label_shift = 0
            global_question_representations, global_context_representations, global_labels = [], [], []
            gatherers = zip(question_representations_gatherer,
                            context_representations_gatherer, labels_gatherer)
            for i, (received_question_representations, received_context_representations, received_labels) in enumerate(gatherers):
                # receiving representations from other GPUs
                if i != self.args.rank:
                    global_question_representations.append(
                        received_question_representations.to(local_question_representations.device))
                    global_context_representations.append(
                        received_context_representations.to(local_context_representations.device))
                    # labels are defined at the batch-level so we need to shift them when concatening batches
                    received_labels[received_labels !=
                                    self.loss_fct.ignore_index] += label_shift
                    # 2N
                    label_shift += received_context_representations.shape[0]
                    global_labels.append(
                        received_labels.to(local_labels.device))
                # keep local representation
                else:
                    global_question_representations.append(
                        local_question_representations)
                    global_context_representations.append(
                        local_context_representations)
                    # labels are defined at the batch-level so we need to shift them when concatening batches
                    local_labels[local_labels !=
                                 self.loss_fct.ignore_index] += label_shift
                    label_shift += local_context_representations.shape[0]  # 2N
                    global_labels.append(local_labels)
            global_question_representations = torch.cat(
                global_question_representations, dim=0)
            global_context_representations = torch.cat(
                global_context_representations, dim=0)
            global_labels = torch.cat(global_labels, dim=0)
        else:
            # (N, d)
            global_question_representations = local_question_representations
            # (2N, d)
            global_context_representations = local_context_representations
            global_labels = local_labels  # N

        # compute similarity
        # (N, 2N)
        similarities = global_question_representations @ global_context_representations.T
        log_probs = self.log_softmax(similarities)

        return self.loss_fct(log_probs, global_labels)


def main_worker(config_training, datasets_config, args):
    print(
        f'Process Launching at GPU {config_training.local_rank} and rank is {config_training.rank}')

    if config_training.distributed and not dist.is_initialized():
        dist.init_process_group(
            backend='nccl', world_size=config_training.world_size, rank=config_training.rank)
    if config_training.distributed:
        torch.device("cuda", index=config_training.local_rank)
    else:
        torch.device("cpu")

    verbose = True
    if config_training.distributed:
        if config_training.rank != 0:
            verbose = False
    if verbose:
        print(f"World size : {config_training.world_size}")

    if config_training.train:
        training_loaders = []
        validation_loaders = []
        for task, args in datasets_config.items():
            validation_loaders.append(
                get_loader(
                    task=task,
                    mode="eval",
                    seed=config_training.seed,
                    workers=config_training.num_workers,
                    verbose=verbose,
                    split="validation",
                    distributed=config_training.distributed,
                    **args
                )
            )
            training_loaders.append(
                get_loader(
                    task=task,
                    mode="eval",
                    split="train",
                    seed=config_training.seed,
                    workers=config_training.num_workers,
                    verbose=verbose,
                    distributed=config_training.distributed,
                    **args
                )
            )
        val_loader = get_val_loader(validation_loaders)
        train_loader = get_multitask_loader(training_loaders, verbose)
        test_loader = None
    elif config_training.test:
        pass
    config_encoder_question = Config.load_json(args.encoder_question_path)
    config_encoder_passage = Config.load_json(args.encoder_passage_path)
    config_model = Config.load_json(args.model_path)

    trainer = Trainer(
        config_question=config_encoder_question,
        config_passage=config_encoder_passage,
        config_model=config_model,
        config_training=config_training,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        train=config_training.train
    )

    if config_training.train:
        trainer.train()
    elif config_training.test:
        trainer.test()


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
    with open(args.training_path, 'r') as f:
        config_training_json = json.load(f)
    datasets_config = config_training_json.pop("datasets")
    config_training_json['tasks'] = [t for t in datasets_config.keys()]
    config_training = Config(config_training_json)
    config_training.world_size = world_size
    config_training.rank = rank
    config_training.local_rank = local_rank
    if world_size > 1:
        config_training.distributed = True
        config_training.multiGPU = True
    else:
        config_training.distributed = False
        config_training.multiGPU = False

    main_worker(config_training, datasets_config, args)

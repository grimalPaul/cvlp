import argparse
import re
from torch import nn
import torch
from cvlep.VLT5.utils import LossMeter, load_state_dict
from pprint import pprint
import os
import logging
from packaging import version
from torch.nn.parallel import DistributedDataParallel as DDP
from torch import distributed as dist
from cvlep.CLIPT5.modeling_t5 import JointEncoder as encoderVLT5
from cvlep.VLT5.param import Config
from cvlep.CLIPT5 import modeling_t5
from cvlep.modeling_cvlp import CVLEP
from cvlep.utils import set_global_logging_level
import random
import numpy as np
from cvlep.CLIPT5.adapters import (
    AdapterLayer,
    AdapterController,
    OutputParallelAdapterLayer,
    MetaAdapterConfig,
    AdapterConfig,
    CompactorConfig,
    LRAdapterConfig,
    TaskEmbeddingController,
    AdapterLayersHyperNetController,
    AdapterLayersOneHyperNetController
)
from cvlep.CLIPT5.prompt import EncoderPromptConfig, DecoderPromptConfig, PromptController
from cvlep.CLIPT5.lora import LoraConfig
from cvlep.CLIPT5.vis_encoder import CLIPResNetEncoder
from cvlep.CLIPT5.clip.model import VisualAdapter
from transformers.models.t5.modeling_t5 import T5LayerNorm
from tqdm import tqdm
from cvlep.viquae_data import get_loader
from torch.utils.tensorboard import SummaryWriter
from cvlep.utils import retrieval

# Check if Pytorch version >= 1.6 to switch between Native AMP and Apex
if version.parse(torch.__version__) < version.parse("1.6"):
    raise NotImplementedError("We did not implement apex")
else:
    _use_native_amp = True
    from torch.cuda.amp import autocast


class Trainer(object):
    def __init__(self, config_question, config_passage, config_model, config_training, train_loader=None, val_loader=None, test_loader=None, train=True, local=False):

        # config of the two part of the model
        config_encoder_question = config_question
        config_encoder_passage = config_passage

        # train config
        self.args = config_training

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

        self.verbose = True
        if self.args.distributed:
            if self.args.rank != 0:
                self.verbose = False
        if train and self.verbose:
            self.writer = SummaryWriter(log_dir=self.args.log_tensorboard_path)

        if not self.verbose:
            set_global_logging_level(logging.ERROR, ["transformers"])

        # create config
        ModelQuestionConfig = self.create_config(config_encoder_question)
        ModelPassageConfig = self.create_config(config_encoder_passage)

        # create tokenizer
        # TODO: we will use the same tokenier for question and answer
        self.tokenizer_question = self.create_tokenizer(
            config_encoder_question)
        self.tokenizer_passage = self.create_tokenizer(config_encoder_passage)

        self.encoder_question = self.create_encoder(ModelQuestionConfig)
        self.encoder_passage = self.create_encoder(ModelPassageConfig)

        if 't5' in config_encoder_question.tokenizer:
            self.encoder_question.resize_token_embeddings(
                self.tokenizer_question.vocab_size)

        if 't5' in config_encoder_passage.tokenizer:
            self.encoder_passage.resize_token_embeddings(
                self.tokenizer_passage.vocab_size)

        # Load Checkpoint encoder question
        self.start_epoch = None
        if config_encoder_question.load_path is not None:
            ckpt_path = config_encoder_question.load_path + '.pth'
            self.load_checkpoint(ckpt_path, "question")

        if config_encoder_question.from_scratch:
            self.init_weights("question")

        # Load Checkpoint encoder passage
        if config_encoder_passage.load_path is not None:
            ckpt_path = config_encoder_passage.load_path + '.pth'
            self.load_checkpoint(ckpt_path, "passage")

        if config_encoder_passage.from_scratch:
            self.init_weights("passage")

        # Create the model
        self.model = self.create_model(config_model)

        if not local:
            # GPU Options
            self.model.to(self.args.local_rank)
            print(f'Model Launching at GPU {self.args.local_rank}')

            # freeze whole parameters first
            self.freeze_whole_model(self.model.image_passage_encoder)
            self.freeze_whole_model(self.model.image_question_encoder)

            # unfreeze selected parameters
            self.unfreeze_parameters(self.model.image_passage_encoder)
            self.unfreeze_parameters(self.model.image_question_encoder)

            self.log_softmax = nn.LogSoftmax(1)
            self.loss_fct = nn.NLLLoss(reduction='mean')

            if train:
                self.optim, self.lr_scheduler = self.create_optimizer_and_scheduler()
                if self.args.fp16 and _use_native_amp:
                    self.scaler = torch.cuda.amp.GradScaler()

            # on instantie plusieurs modèle que l'on va placer sur tel ou tel rank
            # cela veut dire sur tel ou tel gpu
            if self.args.multiGPU:
                if self.args.distributed:
                    self.model = DDP(self.model, device_ids=[
                        config_training.local_rank])

    def train(self):
        if self.verbose:
            best_valid = 0.
            best_epoch = 0
        if self.args.distributed and self.val_loader is not None:
            # to always have the same validation loader
            self.val_loader.sampler.set_epoch(0)

        for epoch in tqdm(range(self.args.epochs)):
            # if self.verbose:
            #     loss_meter = LossMeter()
            #     pbar = tqdm(total=len(self.train_loader), ncols=100)
            # self.model.train()
            # if self.args.distributed:
            #     self.train_loader.sampler.set_epoch(epoch)
            # for step_i, batch in enumerate(self.train_loader):
            #     if self.args.fp16 and _use_native_amp:
            #         with autocast():
            #             if self.args.distributed:
            #                 loss = self.compute_loss(batch)
            #     else:
            #         loss = self.compute_loss(batch)

            #     # loss.backward
            #     if self.args.fp16 and _use_native_amp:
            #         self.scaler.scale(loss).backward()
            #     else:
            #         loss.backward()

            #     loss = loss.detach()
            #     # Update Parameters
            #     if self.args.clip_grad_norm > 0:
            #         if self.args.fp16 and _use_native_amp:
            #             self.scaler.unscale_(self.optim)
            #             torch.nn.utils.clip_grad_norm_(
            #                 self.model.parameters(), self.args.clip_grad_norm)
            #         else:
            #             torch.nn.utils.clip_grad_norm_(
            #                 self.model.parameters(), self.args.clip_grad_norm)

            #     # optim step
            #     update = True
            #     if self.args.gradient_accumulation_steps > 1:
            #         if step_i == 0:
            #             update = False
            #         elif step_i % self.args.gradient_accumulation_steps == 0 or step_i == len(self.train_loader) - 1:
            #             update = True
            #         else:
            #             update = False

            #     if update:
            #         if self.args.fp16 and _use_native_amp:
            #             self.scaler.step(self.optim)
            #             self.scaler.update()
            #         else:
            #             self.optim.step()

            #         if self.lr_scheduler:
            #             self.lr_scheduler.step()
            #         for param in self.model.parameters():
            #             param.grad = None

            #     # Scheduler
            #     if self.lr_scheduler:
            #         if version.parse(torch.__version__) >= version.parse("1.4"):
            #             lr = self.lr_scheduler.get_last_lr()[0]
            #         else:
            #             lr = self.lr_scheduler.get_lr()[0]
            #     else:
            #         try:
            #             lr = self.optim.get_lr()[0]
            #         except AttributeError:
            #             lr = self.args.lr

            #     if self.verbose:
            #         loss_meter.update(loss.item())
            #         desc_str = f'Epoch {epoch} | LR {lr:.6f}'
            #         desc_str += f' | Loss {loss_meter.val:4f}'
            #         pbar.set_description(desc_str)
            #         pbar.update(1)

            # if self.args.distributed:
            #     dist.barrier()

            # if self.verbose:
            #     pbar.close()
            #     self.writer.add_scalar('Training_loss', loss_meter.val, epoch)
            #     self.writer.add_scalar('lr', lr, epoch)
            #     self.writer.flush()
            # Validation
            if self.val_loader is not None:
                self.model.eval()
                with torch.no_grad():
                    if self.verbose:
                        loss_meter = LossMeter()
                        pbar = tqdm(total=len(self.val_loader), ncols=100)
                        all_probs = []
                        all_labels = []
                    for step_i, batch in enumerate(self.val_loader):
                        if self.args.fp16 and _use_native_amp:
                            with autocast():
                                if self.args.distributed:
                                    loss, outputs = self.compute_loss(
                                        batch, return_outputs=True)
                        else:
                            loss, outputs = self.compute_loss(
                                batch, return_outputs=True)
                        if self.verbose:
                            all_probs.append(outputs['log_probs'])
                            all_labels.append(outputs['label_ids'])
                            loss_meter.update(loss.item())
                            desc_str = f'Validation {epoch} | Loss {loss_meter.val:4f}'
                            pbar.set_description(desc_str)
                            pbar.update(1)
                    if self.verbose:
                        all_probs = torch.cat(all_probs, dim=0)
                        all_labels = torch.cat(all_labels, dim=0)
                        eval_prediction = dict(
                            predictions=all_probs, label_ids=all_labels)
                        metrics = retrieval(eval_prediction)
                        pbar.close()
                        self.writer.add_scalar(
                            'Validation_loss', loss_meter.val, epoch)
                        self.writer.add_scalar(
                            "MRR@N*M", metrics['MRR@N*M'], epoch)
                        self.writer.add_scalar(
                            "Hits@1", metrics['hits@1'], epoch)
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
                    log_str += f"\nMRR {metrics['MRR@N*M']} Hits@1: {metrics['hits@1']}"
                    print(log_str)

            if self.args.distributed:
                dist.barrier()

        if self.verbose:
            self.writer.close()

    def test(self):
        if self.test_loader is not None:
            if self.args.distributed:
                self.test_loader.sampler.set_epoch(0)
            self.model.eval()
            if self.verbose:
                loss_meter = LossMeter()
                pbar = tqdm(total=len(self.test_loader), ncols=100)
            for step_i, batch in enumerate(self.test_loader):
                if self.args.fp16 and _use_native_amp:
                    with autocast():
                        if self.args.distributed:
                            loss = self.compute_loss(batch)
                else:
                    loss = self.compute_loss(batch)
                if self.verbose:
                    loss_meter.update(loss.item())
                    desc_str = f'Test Loss {loss_meter.val:4f}'
                    pbar.set_description(desc_str)
                    pbar.update(1)
            if self.verbose:
                pbar.close()

    def compute_loss(self, batch, return_outputs=False):
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
        loss = self.loss_fct(log_probs, global_labels)
        return (loss, dict(log_probs=log_probs, label_ids=global_labels)) if return_outputs else loss

    def create_config(self, config_model):
        from transformers import T5Config

        if 't5' in config_model.backbone:
            config_class = T5Config
        else:
            return None

        config_encoder = config_class.from_pretrained(config_model.backbone)
        args = config_model

        config_encoder._name_or_path = args.backbone
        config_encoder.feat_dim = args.feat_dim
        config_encoder.pos_dim = args.pos_dim
        config_encoder.n_images = 2

        config_encoder.n_boxes = args.n_boxes  # VL adapter

        config_encoder.add_projectionHead = args.add_projectionHead
        if config_encoder.add_projectionHead:
            config_encoder.dim_projectionHead = args.dim_projectionHead

        # for ExpandVisualEmbedding
        config_encoder.expand_vis_embedding = args.expand_vis_embedding  # VL adapter
        config_encoder.n_image_tokens = args.n_image_tokens  # VL adapter

        config_encoder.vis_use_transformer = args.vis_use_transformer  # only for clip BART

        # choose between OneDDownsample, Downsample, SparseSample for the model
        config_encoder.downsample = args.downsample
        config_encoder.oneddownsample = args.oneddownsample
        config_encoder.sparse_sample = args.sparse_sample

        # augment number of layers for projection
        config_encoder.additional_visual_embedding_layers = args.additional_visual_embedding_layers

        # strategy when forwarding with the vision model
        config_encoder.vis_pooling_output = args.vis_pooling_output
        # TODO: Idk for now the use (dont use with T5 just with Bart)
        config_encoder.use_lm_head_adapter = args.use_lm_head_adapter

        # which type of Adapter module or lra
        config_encoder.use_hyperformer = args.use_hyperformer
        config_encoder.use_adapter = args.use_adapter
        config_encoder.use_compacter = args.use_compacter
        config_encoder.use_lradapter = args.use_lradapter
        # reduction factor in adapters
        config_encoder.reduction_factor = args.reduction_factor

        # TODO: use ? add adapter cross attention
        config_encoder.add_adapter_cross_attn = args.add_adapter_cross_attn

        tasks = re.split("[, ]+", args.tasks)  # tranform to list

        # define adapters module
        if args.use_hyperformer or args.use_adapter or args.use_compacter or args.use_lradapter:

            assert config_encoder.use_hyperformer + config_encoder.use_adapter + config_encoder.use_compacter + \
                config_encoder.use_lradapter <= 1, "You can only at most one kind of adapters."

            if args.use_hyperformer:
                CONFIG_CLASS = MetaAdapterConfig
            elif args.use_adapter:
                CONFIG_CLASS = AdapterConfig
            elif args.use_compacter:
                CONFIG_CLASS = CompactorConfig
            elif args.use_lradapter:
                CONFIG_CLASS = LRAdapterConfig

            config_encoder.adapter_config = CONFIG_CLASS()
            config_encoder.adapter_config.tasks = tasks
            config_encoder.adapter_config.input_dim = config_encoder.d_model  # for hyperformer
            # for adapter and compactor
            config_encoder.adapter_config.d_model = config_encoder.d_model
            config_encoder.adapter_config.unique_hyper_net = args.unique_hyper_net
            config_encoder.adapter_config.efficient_unique_hyper_net = args.efficient_unique_hyper_net
            config_encoder.adapter_config.use_single_adapter = args.use_single_adapter
            config_encoder.adapter_config.hypercomplex_division = args.hypercomplex_division
            config_encoder.adapter_config.phm_rank = args.phm_rank
            config_encoder.adapter_config.shared_phm_rule = args.shared_phm_rule
            config_encoder.adapter_config.factorized_phm = args.factorized_phm
            config_encoder.adapter_config.low_rank_rank = args.low_rank_rank
            config_encoder.adapter_config.phm_init_range = args.phm_init_range

            config_encoder.adapter_config.share_down_sampler = args.share_down_sampler
            config_encoder.adapter_config.share_up_sampler = args.share_up_sampler
            config_encoder.adapter_config.reduction_factor = args.reduction_factor
            config_encoder.adapter_config.shared_phm_rule_over_tasks = args.shared_phm_rule_over_tasks

            config_encoder.adapter_config.add_layer_norm_before_adapter = args.add_layer_norm_before_adapter
            config_encoder.adapter_config.add_layer_norm_after_adapter = args.add_layer_norm_after_adapter

            config_encoder.adapter_config.track_z = args.track_z

            if args.projected_task_embedding_dim != -1:
                config_encoder.adapter_config.projected_task_embedding_dim = args.projected_task_embedding_dim
        else:
            config_encoder.adapter_config = None

        # for prompt
        # dimension middle in the linear config of the model for prompt
        config_encoder.mid_dim = args.mid_dim

        if args.encoder_prompt_len > 0:
            config_encoder.encoder_prompt_config = EncoderPromptConfig()
            config_encoder.encoder_prompt_config.prompt_len = args.encoder_prompt_len
            config_encoder.encoder_prompt_config.tasks = tasks
            config_encoder.encoder_prompt_config.use_single_prompt = args.use_single_prompt
            config_encoder.encoder_prompt_config.mid_dim = args.mid_dim
        else:
            config_encoder.encoder_prompt_config = None

        config_encoder.decoder_prompt_config = None

        # for lora
        if args.use_lora:
            config_encoder.lora_config = LoraConfig()
            config_encoder.lora_config.lora_dim = args.lora_dim
            config_encoder.lora_config.lora_alpha = args.lora_alpha
            config_encoder.lora_config.tasks = tasks
            config_encoder.lora_config.use_single_lora = args.use_single_lora

        # VLT5/BART
        config_encoder.use_vis_order_embedding = args.use_vis_order_embedding
        config_encoder.dropout_rate = args.dropout
        config_encoder.dropout = args.dropout
        config_encoder.attention_dropout = args.dropout
        config_encoder.activation_dropout = args.dropout
        config_encoder.use_vis_layer_norm = args.use_vis_layer_norm
        config_encoder.individual_vis_layer_norm = args.individual_vis_layer_norm
        #config_encoder.losses = args.losses
        config_encoder.share_vis_lang_layer_norm = args.share_vis_lang_layer_norm

        # unfreeze or freeze
        config_encoder.use_lora = args.use_lora
        config_encoder.unfreeze_visual_embedding = args.unfreeze_visual_embedding
        config_encoder.encoder_prompt_len = args.encoder_prompt_len
        config_encoder.decoder_prompt_len = args.decoder_prompt_len
        config_encoder.unfreeze_language_model = args.unfreeze_language_model
        config_encoder.unfreeze_lm_head = args.unfreeze_lm_head
        config_encoder.unfreeze_vis_encoder = args.unfreeze_vis_encoder
        config_encoder.unfreeze_vis_last_layer = args.unfreeze_vis_last_layer
        config_encoder.use_vis_adapter = args.use_vis_adapter
        config_encoder.unfreeze_layer_norms = args.unfreeze_layer_norms
        config_encoder.unfreeze_batch_norms = args.unfreeze_batch_norms

        return config_encoder

    def create_model(self, config_model=None):
        model = CVLEP(config_model, self.encoder_question,
                      self.encoder_passage)
        return model

    def create_encoder(self, config_model):

        if 't5' in config_model._name_or_path:
            model_class = encoderVLT5

        else:
            raise NotImplementedError(
                "Thys type of encoder is not implemented")

        model_name = config_model._name_or_path

        model = model_class.from_pretrained(
            model_name,
            config=config_model,
        )
        return model

    def create_tokenizer(self, config_model, **kwargs):

        from transformers import T5Tokenizer, T5TokenizerFast
        from cvlep.CLIPT5.tokenization import VLT5Tokenizer, VLT5TokenizerFast

        if 't5' in config_model.tokenizer:
            if config_model.use_vision:
                # tokenizer_class = VLT5Tokenizer
                tokenizer_class = VLT5TokenizerFast
            else:
                # tokenizer_class = T5Tokenizer
                tokenizer_class = T5TokenizerFast
        else:
            raise ValueError('This type of tokenizer is not implemented')

        tokenizer = tokenizer_class.from_pretrained(
            config_model.tokenizer,
            do_lower_case=config_model.do_lower_case,
            **kwargs
        )
        return tokenizer

    def freeze_whole_model(self, model):
        for _, p in model.named_parameters():
            p.requires_grad = False

    def unfreeze_parameters(self, model):
        targets = ["visual_embedding"]
        # unfreeze the parameters in targets anyway
        if model.config.unfreeze_visual_embedding:
            for n, p in model.named_parameters():
                if any(t in n for t in targets):
                    p.requires_grad = True
                    print(f"{n} is trainable...")
                # else:
                #     p.requires_grad = False

        if model.config.unfreeze_language_model:
            targets = ["lm_head", "shared"]
            for n, p in model.named_parameters():
                if any(t in n for t in targets):
                    p.requires_grad = True
                    print(f"{n} is trainable...")
            for name, sub_module in model.named_modules():

                if isinstance(sub_module, (modeling_t5.T5Stack, modeling_t5.JointEncoder)):
                    print(f"{name} is trainable...")
                    # if len(name.split(".")) < 7: # this will not consider layer norms inside adapters then.
                    for param_name, param in sub_module.named_parameters():
                        param.requires_grad = True

        if model.config.unfreeze_lm_head:
            # shared and lm_head share the same weight
            targets = ["lm_head", "shared"]
            for n, p in model.named_parameters():
                if any(t in n for t in targets):
                    p.requires_grad = True
                    print(f"{n} is trainable...")

        if model.config.use_lora:
            targets = ["lora", "bias"]
            for n, p in model.named_parameters():
                if any(t in n for t in targets):
                    p.requires_grad = True
                    print(f"{n} is trainable...")

        for name, sub_module in model.named_modules():
            if model.config.decoder_prompt_len > 0 or model.config.encoder_prompt_len > 0:
                if isinstance(sub_module, (PromptController)):
                    print(f"{name} is trainable...")
                    # if len(name.split(".")) < 7: # this will not consider layer norms inside adapters then.
                    for param_name, param in sub_module.named_parameters():
                        param.requires_grad = True

            if model.config.unfreeze_vis_encoder:
                if isinstance(sub_module, (CLIPResNetEncoder)):
                    print(f"{name} is trainable...")
                    # if len(name.split(".")) < 7: # this will not consider layer norms inside adapters then.
                    for param_name, param in sub_module.named_parameters():
                        param.requires_grad = True

            if model.config.unfreeze_vis_last_layer:
                if "visual.layer4" in name and "visual.layer4." not in name:
                    print(f"{name} is trainable...")
                    # if len(name.split(".")) < 7: # this will not consider layer norms inside adapters then.
                    for param_name, param in sub_module.named_parameters():
                        param.requires_grad = True

            if model.config.use_vis_adapter:
                if isinstance(sub_module, (VisualAdapter)):
                    print(f"{name} is trainable...")
                    # if len(name.split(".")) < 7: # this will not consider layer norms inside adapters then.
                    for param_name, param in sub_module.named_parameters():
                        param.requires_grad = True

            if model.config.unfreeze_layer_norms:
                if isinstance(sub_module, (T5LayerNorm, nn.LayerNorm)):
                    print(f"{name} is trainable...")
                    # if len(name.split(".")) < 7: # this will not consider layer norms inside adapters then.
                    for param_name, param in sub_module.named_parameters():
                        param.requires_grad = True

            if model.config.unfreeze_batch_norms:
                if isinstance(sub_module, (nn.BatchNorm2d)):
                    print(f"{name} is trainable...")
                    # if len(name.split(".")) < 7: # this will not consider layer norms inside adapters then.
                    for param_name, param in sub_module.named_parameters():
                        param.requires_grad = True

            if model.config.use_adapter or model.config.use_compacter or model.config.use_lradapter:
                if isinstance(sub_module, (AdapterController)):
                    print(f"{name} is trainable...")
                    for param_name, param in sub_module.named_parameters():
                        param.requires_grad = True

            if model.config.use_lm_head_adapter:
                if isinstance(sub_module, (OutputParallelAdapterLayer)):
                    print(f"{name} is trainable...")
                    for param_name, param in sub_module.named_parameters():
                        param.requires_grad = True

            if model.config.use_hyperformer:
                if isinstance(sub_module, (TaskEmbeddingController, AdapterLayersHyperNetController, AdapterLayersOneHyperNetController)):
                    print(f"{name} is trainable...")
                    for param_name, param in sub_module.named_parameters():
                        param.requires_grad = True

    def create_optimizer_and_scheduler(self):
        if self.verbose:
            print('Building Optimizer')

        lr_scheduler = None

        from transformers.optimization import AdamW, get_linear_schedule_with_warmup

        no_decay = ["bias", "LayerNorm.weight"]

        if 'adamw' in self.args.optim:
            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                    "weight_decay": 0.0,
                },
            ]
            optim = AdamW(optimizer_grouped_parameters,
                          lr=self.args.lr, eps=self.args.adam_eps)

        else:
            raise NotImplementedError(
                "We do not implement with other optimizer")

        batch_per_epoch = len(self.train_loader)
        t_total = batch_per_epoch // self.args.gradient_accumulation_steps * self.args.epochs
        warmup_ratio = self.args.warmup_ratio
        warmup_iters = int(t_total * warmup_ratio)
        if self.verbose:
            print("Batch per epoch: %d" % batch_per_epoch)
            print("Total Iters: %d" % t_total)
            print('Warmup ratio:', warmup_ratio)
            print("Warm up Iters: %d" % warmup_iters)

        lr_scheduler = get_linear_schedule_with_warmup(
            optim, warmup_iters, t_total)

        return optim, lr_scheduler

    def load_checkpoint(self, ckpt_path, encoder):
        state_dict = load_state_dict(ckpt_path, 'cpu')
        original_keys = list(state_dict.keys())
        for key in original_keys:
            if key.startswith("vis_encoder."):
                new_key = 'encoder.' + key[len("vis_encoder."):]
                state_dict[new_key] = state_dict.pop(key)

            if key.startswith("model.vis_encoder."):
                new_key = 'model.encoder.' + key[len("model.vis_encoder."):]
                state_dict[new_key] = state_dict.pop(key)
        if encoder == "question":
            results = self.encoder_question.load_state_dict(
                state_dict, strict=False)
        elif encoder == "passage":
            results = self.encoder_passage.load_state_dict(
                state_dict, strict=False)
        if self.verbose:
            print('Model loaded from ', ckpt_path)
            pprint(results)

    def init_weights(self, encoder):
        def init_bert_weights(module):
            """ Initialize the weights."""
            if isinstance(module, (nn.Linear, nn.Embedding)):
                # Slightly different from the TF version which uses truncated_normal for initialization
                # cf https://github.com/pytorch/pytorch/pull/5617
                module.weight.data.normal_(mean=0.0, std=1)
            elif isinstance(module, nn.LayerNorm):
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        if encoder == "question":
            self.encoder_question.apply(init_bert_weights)
            self.encoder_question.init_weights()
        elif encoder == "passage":
            self.encoder_passage.apply(init_bert_weights)
            self.encoder_passage.init_weights()

    def save(self, name):
        if not os.path.isdir(self.args.output):
            os.makedirs(self.args.output, exist_ok=True)
        if self.args.distributed:
            # save image question encoder
            torch.save(
                self.model.module.image_question_encoder.state_dict(),
                os.path.join(self.args.output, f"{name}_question.pth")
            )
            # save image passage encoder
            torch.save(
                self.model.module.image_passage_encoder.state_dict(),
                os.path.join(self.args.output, f"{name}_passage.pth")
            )
        else:
            # save image question encoder
            torch.save(
                self.model.image_question_encoder.state_dict(),
                os.path.join(self.args.output, f"{name}_question.pth")
            )
            # save image passage encoder
            torch.save(
                self.model.image_passage_encoder.state_dict(),
                os.path.join(self.args.output, f"{name}_passage.pth")
            )

    def load(self, epoch):
        path_question = os.path.join(
            self.args.output, f"best_{epoch}_question.pth")
        path_passage = os.path.join(
            self.args.output, f"best_{epoch}_passage.pth")
        if self.args.distributed:
            loc = self.args.rank
        state_dict_question = torch.load(path_question, map_location=loc)
        state_dict_passage = torch.load(path_passage, map_location=loc)

        results_question = self.model.image_passage_encoder.load_state_dict(
            state_dict_passage, strict=False)
        results_passage = self.model.image_question_encoder.load_state_dict(
            state_dict_question, strict=False)

        if self.verbose:
            print(f'Model loaded from \n -{path_question}\n-{path_passage}')
            pprint(results_question)
            pprint(results_passage)

    def embedding_passage(self, batch):
        return self.model.embed_image_passage(batch)

    def embedding_question(self, batch):
        return self.model.embed_image_question(batch)


def main_worker(config_training, args):
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
        print(f'distributed : {config_training.distributed}')

    if config_training.train:
        train_loader = get_loader(
            cls="dpr",
            mode='train',
            batch_size=config_training.batch_size,
            seed=config_training.seed,
            distributed=config_training.distributed,
            workers=config_training.num_workers,
            tokenizer_path=config_training.tokenizer_path,
            dataset_path=config_training.dataset_path,
            kb_path=config_training.kb_path,
            passages_path=config_training.passages_path,
            key_relevant=config_training.key_relevant,
            key_text_question=config_training.key_text_question,
            key_text_passage=config_training.key_text_passage,
            key_vision_features=config_training.key_vision_features,
            key_vision_boxes=config_training.key_vision_boxes,
            split='train',
            key_irrelevant=config_training.key_irrelevant,
            verbose=verbose
        )
        val_loader = get_loader(
            cls="dpr",
            mode='eval',
            batch_size=config_training.valid_batch_size,
            seed=config_training.seed,
            distributed=config_training.distributed,
            workers=config_training.num_workers,
            tokenizer_path=config_training.tokenizer_path,
            dataset_path=config_training.dataset_path,
            kb_path=config_training.kb_path,
            passages_path=config_training.passages_path,
            key_relevant=config_training.key_relevant,
            key_text_question=config_training.key_text_question,
            key_text_passage=config_training.key_text_passage,
            key_vision_features=config_training.key_vision_features,
            key_vision_boxes=config_training.key_vision_boxes,
            split='validation',
            key_irrelevant=config_training.key_irrelevant,
            verbose=verbose
        )

        test_loader = None
    elif config_training.test:
        test_loader = get_loader(
            cls="dpr",
            mode='test',
            batch_size=config_training.test_batch_size,
            seed=config_training.seed,
            distributed=config_training.distributed,
            workers=config_training.num_workers,
            tokenizer_path=config_training.tokenizer_path,
            dataset_path=config_training.dataset_path,
            kb_path=config_training.kb_path,
            passages_path=config_training.passages_path,
            key_relevant=config_training.key_relevant,
            key_text_question=config_training.key_text_question,
            key_text_passage=config_training.key_text_passage,
            key_vision_features=config_training.key_vision_features,
            key_vision_boxes=config_training.key_vision_boxes,
            split='test',
            key_irrelevant=config_training.key_irrelevant,
            verbose=verbose
        )
        train_loader = None
        val_loader = None
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

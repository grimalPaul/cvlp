from distutils.command.config import config
import re
from torch import nn
import torch
from cvlep.VLT5.utils import load_state_dict
from pprint import pprint
import os
import logging
from packaging import version
from torch.nn.parallel import DistributedDataParallel as DDP

from cvlep.CLIPT5.modeling_t5 import VLT5
from cvlep.CLIPT5.modeling_bart import VLBart
from cvlep.VLT5.param import Config

from cvlep.CLIPT5 import modeling_bart, modeling_t5

from cvlep.modeling_cvlp import CVLEP
from cvlep.utils import device
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

class Trainer(object):
    def __init__(self, config_question_path, config_passage_path, config_training_path, train_loader=None, val_loader=None, test_loader=None, train=True):
        # on aura deux configs car deux encoders
        # on fait via deux fichiers séparés
        # on ajoute si vl adapter ou non pour nous faciliter les choses

        # trois fichier
        # config de chaque encoder
        # config du training
        config_encoder_question = Config.load_json(config_question_path)
        config_encoder_passage = Config.load_json(config_passage_path)
        config_training = Config.load_json(config_training_path)

        self.args = config_training

        # Set seeds
        torch.manual_seed(self.args.seed)
        random.seed(self.args.seed)
        np.random.seed(self.args.seed)
        self.verbose = True

        # TODO:apply for contrastive learning
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

        # create config
        ModelQuestionConfig = self.create_config(config_encoder_question)
        ModelPassageConfig = self.create_config(config_encoder_passage)

        # create tokenizer
        # TODO: we will use the same tokenier for question and answer        
        self.tokenizer_question = self.create_tokenizer(
            config_encoder_question)
        self.tokenizer_passage = self.create_tokenizer(config_encoder_passage)

        # modify tokenizer
        if 'bart' in config_encoder_question.tokenizer:
            if ModelQuestionConfig.use_vis_order_embedding:
                additional_special_tokens = [f'<extra_id_{i}>' for i in range(100-1, -1, -1)] + \
                    [f'<vis_extra_id_{i}>' for i in range(100-1, -1, -1)]
                special_tokens_dict = {
                    'additional_special_tokens': additional_special_tokens}
                num_added_toks_question = self.tokenizer_question.add_special_tokens(
                    special_tokens_dict)
                ModelQuestionConfig.default_obj_order_ids = self.tokenizer_question.convert_tokens_to_ids(
                    [f'<vis_extra_id_{i}>' for i in range(100)])

        if 'bart' in config_encoder_passage.tokenizer:
            if ModelPassageConfig.use_vis_order_embedding:
                additional_special_tokens = [f'<extra_id_{i}>' for i in range(100-1, -1, -1)] + \
                    [f'<vis_extra_id_{i}>' for i in range(100-1, -1, -1)]
                special_tokens_dict = {
                    'additional_special_tokens': additional_special_tokens}
                num_added_toks_passage = self.tokenizer_passage.add_special_tokens(
                    special_tokens_dict)
                ModelPassageConfig.default_obj_order_ids = self.tokenizer_passage.convert_tokens_to_ids(
                    [f'<vis_extra_id_{i}>' for i in range(100)])

        self.encoder_question = self.create_encoder(ModelQuestionConfig)
        self.encoder_passage = self.create_encoder(ModelPassageConfig)
        
        if 't5' in config_encoder_question.tokenizer:
            self.encoder_question.resize_token_embeddings(
                self.tokenizer_question.vocab_size)
        elif 'bart' in config_encoder_question.tokenizer:
            self.encoder_question.resize_tokCVLPDen_embeddings(
                self.encoder_question.model.shared.num_embeddings + num_added_toks_question)
        
        
        if 't5' in config_encoder_passage.tokenizer:
            self.encoder_passage.resize_token_embeddings(
                self.tokenizer_passage.vocab_size)
        elif 'bart' in config_encoder_passage.tokenizer:
            self.encoder_passage.resize_token_embeddings(
                self.encoder_passage.model.shared.num_embeddings + num_added_toks_passage)
        
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
        self.model = self.create_model()

        # freeze whole parameters first
        self.freeze_whole_model(self.model.image_passage_encoder) 
        self.freeze_whole_model(self.model.image_question_encoder)
        
        # unfreeze selected parameters
        self.unfreeze_parameters(self.model.image_passage_encoder) 
        self.unfreeze_parameters(self.model.image_question_encoder)

        self.model.to(device)
        # pas utile les vocsize ont été enregistrés dans les modèles mais on pourra
        # etre amené à devoir les changer

        # Normalement déjà dans la confiq des encoders que l'on a enregistré
        


        """
         if self.args.distributed:
            if self.args.gpu != 0:
                self.args.verbose = False

        if not self.args.verbose:
            set_global_logging_level(logging.ERROR, ["transformers"])
        # GPU Options
        print(f'Model Launching at GPU {self.args.gpu}')
        if self.verbose:
            from time import time
            start = time()
        self.model = self.model.to(args.gpu)

        # Optimizer
        if train:
            self.optim, self.lr_scheduler = self.create_optimizer_and_scheduler()

            if self.args.fp16 and _use_native_amp:
                self.scaler = torch.cuda.amp.GradScaler()
            elif _use_apex:
                self.model, self.optim = amp.initialize(
                    self.model, self.optim, opt_level='O1', verbosity=self.verbose)

        if args.multiGPU:
            if args.distributed:
                self.model = DDP(self.model, device_ids=[args.gpu],
                                 find_unused_parameters=True
                                 )
        if self.verbose:
            print(f'It took {time() - start:.1f}s')"""

    def create_config(self, config_model):
        from transformers import T5Config, BartConfig

        if 't5' in config_model.backbone:
            config_class = T5Config
        elif 'bart' in config_model.backbone:
            config_class = BartConfig
        else:
            return None

        config_encoder = config_class.from_pretrained(config_model.backbone)
        args = config_model

        config_encoder._name_or_path = args.backbone
        config_encoder.feat_dim = args.feat_dim
        config_encoder.pos_dim = args.pos_dim
        config_encoder.n_images = 2

        config_encoder.n_boxes = args.n_boxes # VL adapter
        
        # for ExpandVisualEmbedding
        config_encoder.expand_vis_embedding = args.expand_vis_embedding # VL adapter
        config_encoder.n_image_tokens = args.n_image_tokens # VL adapter

        config_encoder.vis_use_transformer = args.vis_use_transformer # only for clip BART

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

        tasks = re.split("[, ]+", args.tasks) # tranform to list

        # define adapters module
        if args.use_hyperformer or args.use_adapter or args.use_compacter or args.use_lradapter:
            
            assert config_encoder.use_hyperformer + config_encoder.use_adapter + config_encoder.use_compacter + config_encoder.use_lradapter <= 1, "You can only at most one kind of adapters."

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
            config_encoder.adapter_config.input_dim = config_encoder.d_model # for hyperformer
            config_encoder.adapter_config.d_model = config_encoder.d_model # for adapter and compactor
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

        if args.decoder_prompt_len > 0:
            config_encoder.decoder_prompt_config = DecoderPromptConfig()
            config_encoder.decoder_prompt_config.prompt_len = args.decoder_prompt_len
            config_encoder.decoder_prompt_config.tasks = tasks
            config_encoder.decoder_prompt_config.use_single_prompt = args.use_single_prompt
            config_encoder.decoder_prompt_config.mid_dim = args.mid_dim
        else:
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
        
        # TODO: add a way to use only the decoder
        config_encoder.embed_with_decoder = True
        
        # unfreeze or freeze
        config_encoder.unfreeze_language_model=args.unfreeze_language_model
        config_encoder.unfreeze_lm_head=args.unfreeze_lm_head
        config_encoder.unfreeze_vis_encoder=args.unfreeze_vis_encoder
        config_encoder.unfreeze_vis_last_layer=args.unfreeze_vis_last_layer
        config_encoder.use_vis_adapter=args.use_vis_adapter
        config_encoder.unfreeze_layer_norms=args.unfreeze_layer_norms
        config_encoder.unfreeze_batch_norms=args.unfreeze_batch_normm

        return config_encoder

    def create_model(self, config_model=None):
        model = CVLEP(config_model, self.encoder_question,
                      self.encoder_passage)
        return model

    def create_encoder(self, config_model):

        if 't5' in config_model._name_or_path:
            model_class = VLT5

        elif 'bart' in config_model._name_or_path:
            model_class = VLBart
            
        model_name = config_model._name_or_path
        
        model = model_class.from_pretrained(
            model_name,
            config=config_model,
            )
        return model

    def create_tokenizer(self, config_model, **kwargs):

        from transformers import T5Tokenizer, BartTokenizer, T5TokenizerFast, BartTokenizerFast
        from cvlep.VLT5.tokenization import VLT5Tokenizer, VLT5TokenizerFast

        if 't5' in config_model.tokenizer:
            if config_model.use_vision:
                # tokenizer_class = VLT5Tokenizer
                tokenizer_class = VLT5TokenizerFast
            else:
                # tokenizer_class = T5Tokenizer
                tokenizer_class = T5TokenizerFast
        elif 'bart' in config_model.tokenizer:
            tokenizer_class = BartTokenizer
            # tokenizer_class = BartTokenizerFast
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
        for n, p in model.named_parameters():
            if any(t in n for t in targets):
                p.requires_grad = True
                print(f"{n} is trainable...")
            # else:
            #     p.requires_grad = False

        if self.args.unfreeze_language_model:
            targets = ["lm_head", "shared"]
            for n, p in model.named_parameters():
                if any(t in n for t in targets):
                    p.requires_grad = True
                    print(f"{n} is trainable...")
            for name, sub_module in model.named_modules():
                
                if isinstance(sub_module, (modeling_bart.JointEncoder, modeling_bart.BartDecoder, modeling_t5.T5Stack, modeling_t5.JointEncoder)):
                    print(f"{name} is trainable...")
                    # if len(name.split(".")) < 7: # this will not consider layer norms inside adapters then.
                    for param_name, param in sub_module.named_parameters():
                        param.requires_grad = True

        if self.args.unfreeze_lm_head:
            targets = ["lm_head", "shared"] # shared and lm_head share the same weight
            for n, p in model.named_parameters():
                if any(t in n for t in targets):
                    p.requires_grad = True
                    print(f"{n} is trainable...")

        if self.args.use_lora:
            targets = ["lora", "bias"]
            for n, p in model.named_parameters():
                if any(t in n for t in targets):
                    p.requires_grad = True
                    print(f"{n} is trainable...")

        for name, sub_module in model.named_modules():
            if self.args.decoder_prompt_len > 0 or self.args.encoder_prompt_len > 0:
                if isinstance(sub_module, (PromptController)):
                    print(f"{name} is trainable...")
                    # if len(name.split(".")) < 7: # this will not consider layer norms inside adapters then.
                    for param_name, param in sub_module.named_parameters():
                        param.requires_grad = True
        
            if self.args.unfreeze_vis_encoder:
                if isinstance(sub_module, (CLIPResNetEncoder)):
                    print(f"{name} is trainable...")
                    # if len(name.split(".")) < 7: # this will not consider layer norms inside adapters then.
                    for param_name, param in sub_module.named_parameters():
                        param.requires_grad = True

            if self.args.unfreeze_vis_last_layer:
                if "visual.layer4" in name and "visual.layer4." not in name:
                    print(f"{name} is trainable...")
                    # if len(name.split(".")) < 7: # this will not consider layer norms inside adapters then.
                    for param_name, param in sub_module.named_parameters():
                        param.requires_grad = True

            if self.args.use_vis_adapter:
                if isinstance(sub_module, (VisualAdapter)):
                    print(f"{name} is trainable...")
                    # if len(name.split(".")) < 7: # this will not consider layer norms inside adapters then.
                    for param_name, param in sub_module.named_parameters():
                        param.requires_grad = True

            if self.args.unfreeze_layer_norms:
                if isinstance(sub_module, (T5LayerNorm, nn.LayerNorm)):
                    print(f"{name} is trainable...")
                    # if len(name.split(".")) < 7: # this will not consider layer norms inside adapters then.
                    for param_name, param in sub_module.named_parameters():
                        param.requires_grad = True

            if self.args.unfreeze_batch_norms:
                if isinstance(sub_module, (nn.BatchNorm2d)):
                    print(f"{name} is trainable...")
                    # if len(name.split(".")) < 7: # this will not consider layer norms inside adapters then.
                    for param_name, param in sub_module.named_parameters():
                        param.requires_grad = True

            if self.args.use_adapter or self.args.use_compacter or self.args.use_lradapter:
                if isinstance(sub_module, (AdapterController)):
                    print(f"{name} is trainable...")
                    for param_name, param in sub_module.named_parameters():
                        param.requires_grad = True

            if self.args.use_lm_head_adapter:
                if isinstance(sub_module, (OutputParallelAdapterLayer)):
                    print(f"{name} is trainable...")
                    for param_name, param in sub_module.named_parameters():
                        param.requires_grad = True

            if self.args.use_hyperformer:
                if isinstance(sub_module, (TaskEmbeddingController, AdapterLayersHyperNetController, AdapterLayersOneHyperNetController)):
                    print(f"{name} is trainable...")
                    for param_name, param in sub_module.named_parameters():
                        param.requires_grad = True

            return model

    def create_optimizer_and_scheduler(self):
        if self.verbose:
            print('Building Optimizer')

        lr_scheduler = None

        if 'adamw' in self.args.optim:
            from transformers.optimization import AdamW, get_linear_schedule_with_warmup
            batch_per_epoch = len(self.train_loader)
            t_total = batch_per_epoch // self.args.gradient_accumulation_steps * self.args.epochs
            warmup_ratio = self.args.warmup_ratio
            warmup_iters = int(t_total * warmup_ratio)
            if self.verbose:
                print("Batch per epoch: %d" % batch_per_epoch)
                print("Total Iters: %d" % t_total)
                print('Warmup ratio:', warmup_ratio)
                print("Warm up Iters: %d" % warmup_iters)

            no_decay = ["bias", "LayerNorm.weight"]
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
            lr_scheduler = get_linear_schedule_with_warmup(
                optim, warmup_iters, t_total)
        else:
            optim = self.args.optimizer(
                list(self.model.parameters()), self.args.lr)
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

    def predict(self):
        pass

    def evaluate(self):
        pass

    def save(self, name):
        if not os.path.isdir(self.args.output):
            os.makedirs(self.args.output, exist_ok=True)
        torch.save(self.model.state_dict(), os.path.join(
            self.args.output, "%s.pth" % name))

    def load(self, path, loc=None):
        if loc is None and hasattr(self.args, 'gpu'):
            loc = f'cuda:{self.args.gpu}'
        state_dict = torch.load("%s.pth" % path, map_location=loc)

        original_keys = list(state_dict.keys())
        for key in original_keys:
            if key.startswith("module.vis_encoder."):
                new_key = 'module.encoder.' + key[len("module.vis_encoder."):]
                state_dict[new_key] = state_dict.pop(key)

            if key.startswith("module.model.vis_encoder."):
                new_key = 'module.model.encoder.' + \
                    key[len("module.model.vis_encoder."):]
                state_dict[new_key] = state_dict.pop(key)

        results = self.model.load_state_dict(state_dict, strict=False)
        if self.verbose:
            print('Model loaded from ', path)
            pprint(results)

    def embedding_passage(self, **kwargs):
        return self.model.embed_image_passage(**kwargs)

    def embedding_question(self, **kwargs):
        return self.model.embed_image_question(**kwargs)

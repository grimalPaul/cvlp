from distutils.command.config import config
from torch import nn
import torch
from cvlep.VLT5.utils import load_state_dict
from pprint import pprint
import os
from cvlep.utils import get_config, set_global_logging_level, dotdict
import logging
from packaging import version
from torch.nn.parallel import DistributedDataParallel as DDP

from cvlep.VLT5.modeling_t5 import VLT5
from cvlep.VLT5.modeling_bart import VLBart
from cvlep.VLT5.param import Config

from cvlep.modeling_cvlp import CVLEP
from cvlep.utils import device

from transformers import BartConfig


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
        config_encoder.use_vis_order_embedding = args.use_vis_order_embedding
        config_encoder.dropout_rate = args.dropout
        config_encoder.dropout = args.dropout
        config_encoder.attention_dropout = args.dropout
        config_encoder.activation_dropout = args.dropout
        config_encoder.use_vis_layer_norm = args.use_vis_layer_norm
        config_encoder.individual_vis_layer_norm = args.individual_vis_layer_norm
        config_encoder.losses = args.losses
        config_encoder.share_vis_lang_layer_norm = args.share_vis_lang_layer_norm
        # TODO:Fix
        config_encoder.embed_with_decoder = True
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
            max_length=config_model.max_text_length,
            do_lower_case=config_model.do_lower_case,
            **kwargs
        )
        return tokenizer

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


"""
from transformers import Trainer, TrainingArguments

class CVLEP_HFTrainer(Trainer):
    # https://huggingface.co/docs/transformers/main_classes/trainer#trainer 
    # try something like that if available for our model
    # Our model need to always return tupple or subclasses of ModelOutput
    # voir les autres trucs à mettre en place
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute_loss(self,model,inputs, return_outputs = False):
        raise NotImplementedError()

    def create_optimizer_and_scheduler(self):
        raise NotImplementedError()

    def training_step(self):
        raise NotImplementedError()

    def evaluate(self):
        raise NotImplementedError()

    def save_state(self):
        raise NotImplementedError()

    def write_prediction(self):
        raise NotImplementedError()
    
    def write_metrics(self):
        raise NotImplementedError()
"""

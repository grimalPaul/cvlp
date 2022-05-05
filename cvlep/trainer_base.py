from torch import nn
import torch
from cvlep.VLT5.utils import load_state_dict
from pprint import pprint
import os
from cvlep.utils import set_global_logging_level, get_config
import logging
from packaging import version
from torch.nn.parallel import DistributedDataParallel as DDP

from cvlep.VLT5.modeling_t5 import JointEncoder as VLt5Encoder
from cvlep.VLT5.modeling_bart import JointEncoder as VLBartEncoder

from cvlep.modeling_cvlep import CVLEP
_use_native_amp = False
_use_apex = False

# Check if Pytorch version >= 1.6 to switch between Native AMP and Apex
if version.parse(torch.__version__) < version.parse("1.6"):
    from transformers.file_utils import is_apex_available
    if is_apex_available():
        from apex import amp
    _use_apex = True
else:
    _use_native_amp = True
    from torch.cuda.amp import autocast


def get_encoder(config: dict):
    if config.model.backbone == 't5':
        encoder = VLt5Encoder.from_pretrained(
            config.model.pretrained_model_name_or_path)
    elif config.model.backbone == 'bart':
        encoder = VLBartEncoder(config.model.pretrained_model_name_or_path)
    else:
        raise NotImplementedError('This type of encoder is not implemented')
    return encoder


def get_tokenizer(config: dict):
    from transformers import T5Tokenizer, BartTokenizer, T5TokenizerFast, BartTokenizerFast
    from cvlep.VLT5.tokenization import VLT5Tokenizer, VLT5TokenizerFast
    if config.tokenizer.backbone == 't5':
        if config.tokenizer.use_vision:
            # tokenizer_class = VLT5Tokenizer
            tokenizer_class = VLT5TokenizerFast
        else:
            # tokenizer_class = T5Tokenizer
            tokenizer_class = T5TokenizerFast
    elif config.tokenizer.backbone ==  'bart':
            tokenizer_class = BartTokenizer
            # tokenizer_class = BartTokenizerFast
    else:
        raise NotImplementedError('This type of tokenizer is not implemented')
    tokenizer = tokenizer_class.from_pretrained(
        config.tokenizer.pretrained_model_name_or_path,
        max_length=config.max_text_length,
        do_lower_case=config.args.do_lower_case,
        **kwargs
    )
    return tokenizer

class Trainer(object):
    def __init__(self, args, train_loader=None, val_loader=None, test_loader=None, train=True):
        self.args = args

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

        self.verbose = True
        if self.args.distributed:
            if self.args.gpu != 0:
                self.verbose = False

        if self.args.tokenizer is None:
            self.args.tokenizer = self.args.backbone

        if not self.verbose:
            set_global_logging_level(logging.ERROR, ["transformers"])

        from cvlep.modeling_cvlep import CVLEP

        # on va passer la config de l'encoder en argument

        # de base on a un modèle complet mais on veut juste joint encoder
        # si je peux charger encoder depuis pretrained ca peut me permettre de charger directement ce dont j'ai besoin et ne pas passer par les configs compliqués

        config = get_config(args.config_encoder)

        self.model = self.create_model(config)
        self.tokenizer_question, self.tokenizer_passage = self.create_tokenizer(
            config)

        model_kwargs = {}
        if 't5' in args.backbone:
            model_class =
        elif 'bart' in args.backbone:
            model_class = VLBartPretraining

        config = self.create_config()
        self.tokenizer = self.create_tokenizer()
        if 'bart' in self.args.tokenizer:
            num_added_toks = 0
            if config.use_vis_order_embedding:
                additional_special_tokens = [f'<extra_id_{i}>' for i in range(100-1, -1, -1)] + \
                    [f'<vis_extra_id_{i}>' for i in range(100-1, -1, -1)]
                special_tokens_dict = {
                    'additional_special_tokens': additional_special_tokens}
                num_added_toks = self.tokenizer.add_special_tokens(
                    special_tokens_dict)

                config.default_obj_order_ids = self.tokenizer.convert_tokens_to_ids(
                    [f'<vis_extra_id_{i}>' for i in range(100)])

        self.model = self.create_model(model_class, config, **model_kwargs)

        if 't5' in self.args.tokenizer:
            self.model.resize_token_embeddings(self.tokenizer.vocab_size)
        elif 'bart' in self.args.tokenizer:
            self.model.resize_token_embeddings(
                self.model.model.shared.num_embeddings + num_added_toks)

        self.model.tokenizer = self.tokenizer

        # Load Checkpoint
        self.start_epoch = None
        if args.load is not None:
            ckpt_path = args.load + '.pth'
            self.load_checkpoint(ckpt_path)
            self.start_epoch = int(args.load.split('Epoch')[-1])

        if self.args.from_scratch:
            self.init_weights()

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
            print(f'It took {time() - start:.1f}s')

    def create_model(self, config=None, **kwargs):

        question_encoder = get_encoder(config.encoder_question)
        passage_encoder = get_encoder(config.encoder_passage)

        config_model = dict()
        config_model.update(use_projection = config.cvlep.use_projection)
        model = CVLEP(
            config,
            image_question_encoder=question_encoder,
            image_passage_encoder=passage_encoder
        )
        return model

    def create_tokenizer(self, config):
        
        tokenizer_question = get_tokenizer(config.encoder_question)
        tokenizer_passage = get_tokenizer(config.encoder_passage)

        return tokenizer_question,tokenizer_passage

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

    def load_checkpoint(self, ckpt_path):
        state_dict = load_state_dict(ckpt_path, 'cpu')

        original_keys = list(state_dict.keys())
        for key in original_keys:
            if key.startswith("vis_encoder."):
                new_key = 'encoder.' + key[len("vis_encoder."):]
                state_dict[new_key] = state_dict.pop(key)

            if key.startswith("model.vis_encoder."):
                new_key = 'model.encoder.' + key[len("model.vis_encoder."):]
                state_dict[new_key] = state_dict.pop(key)

        results = self.model.load_state_dict(state_dict, strict=False)
        if self.verbose:
            print('Model loaded from ', ckpt_path)
            pprint(results)

    def init_weights(self):

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
        self.model.apply(init_bert_weights)
        self.model.init_weights()

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

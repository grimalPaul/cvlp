"""embed dataset

Usage:

python -m ir.encode_dataset --path_dataset=<path> --type=<question or passage> --path_config_model=<path_to_config_model> \
    --key_text=<key_text> --key_token=<key_token> --which_tokenizer=<tokenizer used>
"""

from cvlep.trainer_base import Trainer
from datasets import load_from_disk, disable_caching
import argparse
import torch
from ir.utils import create_kwargs

disable_caching()

def add_tokenize(item, key_token,tokenizer, key_text, **kwargs):
    item[key_token] = tokenizer(item[key_text], return_tensors='pt',truncation=True, padding=True)
    return item

def tokenize_dataset(path_config_model, which_tokenizer, path_dataset, **kwargs):
    # on tokenize le text dans le dataset
    dataset = load_from_disk(path_dataset)
    trainer = Trainer(path_config_model)
    if which_tokenizer == 'question':
        tokenizer = trainer.tokenizer_question
    elif which_tokenizer == 'passage':
        tokenizer = trainer.tokenizer_question
    else:
        raise NotImplementedError()
    kwargs.update(tokenizer=tokenizer)
    dataset = dataset.map(add_tokenize, fn_kwargs=kwargs)
    dataset.save_to_disk(path_dataset)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_dataset', type=str, required=True)
    parser.add_argument('--type', type=str, required=True)
    parser.add_argument('--which_tokenizer', type=str, required=True)
    parser.add_argument('--path_config_model', type=str, required=True)
    parser.add_argument('--key_text', type=str, required=True)
    parser.add_argument('--key_token', type=str, required=True)
    arg = parser.parse_args()
    kwargs = create_kwargs(arg)
    if arg.type == 'encode_text':
        tokenize_dataset(**kwargs)
    else:
        raise NotImplementedError()
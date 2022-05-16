"""embed dataset

Usage:

python -m embedding_dataset --dataset_path=<path> --type=question --model_config_path=<path_to_config_model> \
    --key_boxes=<key_boxes> --key_vision_features --key_token=<key_token> --key_embedding=<key_embedding>

python -m embedding_dataset --dataset_path=<path> --type=<question or passage> --model_config_path=<path_to_config_model> \
    --key_boxes=<key_boxes> --key_vision_features --key_token=<key_token> --key_embedding=<key_embedding> --kb_path=<kb_path>
"""

from numpy import require
from cvlep.trainer_base import Trainer
from datasets import load_from_disk, disable_caching
import argparse
import torch
from ir.utils import create_kwargs
from cvlep.utils import device
disable_caching()

# surement refactto


def map_embed_question(item, key_boxes: str, key_vision_features: str, key_token: str, key_embedding: str, trainer: Trainer):
    vision_features = torch.Tensor(item[key_vision_features]).to(device)
    boxes = torch.Tensor(item[key_boxes]).to(device)
    inputs_ids = torch.LongTensor(item[key_token]).to(device)
    item[key_embedding] = trainer.embedding_question(
        vis_inputs=(vision_features, boxes),
        input_ids=inputs_ids,
        return_pooled_output=True
    ).pooler_output
    return item


def map_embed_passage(item, key_boxes: str, key_vision_features: str, key_token: str, key_embedding: str, trainer: Trainer, kb):
    kb_index = item['index']
    # get vision embedding from the kb
    vision_features = torch.Tensor(
        kb[kb_index][key_vision_features]).to(device)
    boxes = torch.Tensor(kb[kb_index][key_boxes]).to(device)
    inputs_ids = torch.LongTensor(item[key_token]).to(device)
    item[key_embedding] = trainer.embedding_question(
        vis_inputs=(vision_features, boxes),
        input_ids=inputs_ids,
        return_pooled_output=True
    ).pooler_output
    return item


def dataset_embed_question(dataset_path: str, model_config_path: str, **kwargs):
    dataset = load_from_disk(dataset_path)
    trainer = Trainer(model_config_path)
    kwargs.update(trainer=trainer)
    dataset = dataset.map(map_embed_question,
                          batched=False, fn_kwargs=kwargs)
    dataset.save_to_disk(dataset_path)


def dataset_embed_passage(dataset_path: str,  model_config_path: str, kb_path: str, **kwargs):
    dataset = load_from_disk(dataset_path)
    kb = load_from_disk(kb_path)
    trainer = Trainer(model_config_path)
    kwargs.update(trainer=trainer, kb=kb)
    dataset = dataset.map(map_embed_passage,
                          batched=False, fn_kwargs=kwargs)
    dataset.save_to_disk(dataset_path)


if __name__ == '__main__':
    # we consider that we have all we need to compute the embedding in the dataset
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, required=True)
    parser.add_argument('--type', type=str, required=True)
    parser.add_argument('--model_config_path', type=str, required=True)
    parser.add_argument('--key_vision_features', type=str, required=True)
    parser.add_argument('--key_boxes', type=str, required=True)
    parser.add_argument('--key_token', type=str, required=True)
    parser.add_argument('--key_embedding', type=str, required=True)
    parser.add_argument('--kb_path', type=str, required=False)
    arg = parser.parse_args()
    kwargs = create_kwargs(arg)
    kwargs.pop('type')
    if arg.type == 'question':
        dataset_embed_question(**kwargs)
    elif arg.type == 'passage':
        dataset_embed_passage(**kwargs)
    else:
        raise NotImplementedError()

"""embed dataset

Usage:

python -m embedding_dataset --dataset_path=<path> --type=question --config_question_path=<> \
    --config_passage_path=<> --config_training_path=<> --key_boxes=<key_boxes> --key_vision_features \
        --key_token=<key_token> --key_embedding=<key_embedding>

python -m embedding_dataset --dataset_path=<path> --type=passage -config_question_path=<> \
    --config_passage_path=<>  --config_training_path=<> --key_boxes=<key_boxes> \
        --key_vision_features --key_token=<key_token> --key_embedding=<key_embedding> --kb_path=<kb_path>
"""

from cvlep.trainer_base import Trainer
from datasets import load_from_disk, disable_caching
import argparse
import torch
from processing.utils import create_kwargs
from cvlep.utils import device
import torch.nn.functional as F

disable_caching()


def map_embed_question(item, key_boxes: str, key_vision_features: str, key_text: str, key_embedding: str, method, tokenizer, pool_strategy=None):
    vision_features = torch.Tensor(item[key_vision_features]).to(device)
    boxes = torch.Tensor(item[key_boxes]).to(device)
    vision_features = torch.squeeze(vision_features, dim=1)
    boxes = torch.squeeze(boxes, dim=1)
    input_ids = tokenizer(
        item[key_text], return_tensors='pt', padding=True, truncation=True).input_ids.to(device)
    item[key_embedding] = method(
        vis_inputs=(vision_features, boxes),
        input_ids=input_ids,
        return_pooled_output=True,
        pool_strategy=pool_strategy
    ).pooler_output.cpu().numpy()
    return item


def map_embed_passage(item, key_boxes: str, key_vision_features: str, key_text: str, key_embedding: str, method, kb, tokenizer, pool_strategy=None):
    kb_index = item['index']
    # get vision embedding from the kb
    vision_features = torch.Tensor(
        kb[kb_index][key_vision_features]).to(device)
    boxes = torch.Tensor(kb[kb_index][key_boxes]).to(device)
    vision_features = torch.squeeze(vision_features, dim=1)
    boxes = torch.squeeze(boxes, dim=1)
    input_ids = tokenizer(
        item[key_text], return_tensors='pt', padding=True, truncation=True).input_ids.to(device)
    item[key_embedding] = method(
        vis_inputs=(vision_features, boxes),
        input_ids=input_ids,
        return_pooled_output=True,
        pool_strategy=pool_strategy
    ).pooler_output.cpu().numpy()
    return item


def dataset_embed_question(dataset_path: str,  config_question_path: str, config_passage_path: str, config_training_path: str, batch_size: int, **kwargs):
    dataset = load_from_disk(dataset_path)
    trainer = Trainer(config_question_path,
                      config_passage_path, config_training_path)
    tokenizer = trainer.tokenizer_question
    kwargs.update(method=trainer.embedding_question, tokenizer=tokenizer)
    dataset = dataset.map(map_embed_question,
                          batched=True, batch_size=batch_size, fn_kwargs=kwargs)
    dataset.save_to_disk(dataset_path)


def dataset_embed_passage(dataset_path: str,  config_question_path: str, config_passage_path: str, config_training_path: str, kb_path: str, batch_size: int, **kwargs):
    dataset = load_from_disk(dataset_path)
    kb = load_from_disk(kb_path)
    trainer = Trainer(config_question_path,
                      config_passage_path, config_training_path)
    tokenizer = trainer.tokenizer_passage
    kwargs.update(method=trainer.embedding_passage, kb=kb, tokenizer=tokenizer)
    dataset = dataset.map(map_embed_passage,
                          batched=True, batch_size=batch_size, fn_kwargs=kwargs)
    dataset.save_to_disk(dataset_path)


if __name__ == '__main__':
    # we consider that we have all we need to compute the embedding in the dataset
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, required=True)
    parser.add_argument('--type', type=str, required=True)
    parser.add_argument('--config_question_path', type=str, required=True)
    parser.add_argument('--config_passage_path', type=str, required=True)
    parser.add_argument('--config_training_path', type=str, required=True)
    parser.add_argument('--key_vision_features', type=str, required=True)
    parser.add_argument('--key_boxes', type=str, required=True)
    parser.add_argument('--key_embedding', type=str, required=True)
    parser.add_argument('--key_text', type=str, required=True)
    parser.add_argument('--kb_path', type=str, required=False)
    parser.add_argument('--batch_size', type=int, required=False, default=128)
    parser.add_argument('--pool_strategy', type=str, required=False)
    arg = parser.parse_args()
    kwargs = create_kwargs(arg)
    kwargs.pop('type')
    if arg.type == 'question':
        kwargs.pop('kb_path')
        dataset_embed_question(**kwargs)
    elif arg.type == 'passage':
        dataset_embed_passage(**kwargs)
    else:
        raise NotImplementedError()

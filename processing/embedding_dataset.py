"""embed dataset

Usage:

python -m embedding_dataset --dataset_path=<path> --type=question --config_question_path=<> \
    --config_passage_path=<> --config_training_path=<> --key_boxes=<key_boxes> --key_vision_features \
        --key_token=<key_token> --key_embedding=<key_embedding>

python -m embedding_dataset --dataset_path=<path> --type=passage -config_question_path=<> \
    --config_passage_path=<>  --config_training_path=<> --key_boxes=<key_boxes> \
        --key_vision_features --key_token=<key_token> --key_embedding=<key_embedding> --kb_path=<kb_path>
"""

from cvlep.VLT5.param import Config
from cvlep.trainer_base_vladapter import Trainer
from datasets import load_from_disk, disable_caching
import argparse
import torch
from processing.utils import create_kwargs

disable_caching()


def map_embed_question(item, key_boxes: str, key_vision_features: str, key_text: str, key_embedding: str, method, tokenizer, pool_strategy=None):
    vision_features = torch.Tensor(item[key_vision_features])
    vision_features = torch.squeeze(vision_features, dim=1)
    if key_boxes is None:
        boxes = torch.zeros(
                vision_features.shape[0],vision_features.shape[1], 4) 
    else:
        boxes = torch.Tensor(item[key_boxes])
        boxes = torch.squeeze(boxes, dim=1)
    input = tokenizer(
        item[key_text], return_tensors='pt', padding=True, truncation=True)
    batch = {
            "input_ids":input.input_ids,
            "attention_mask":input.attention_mask,
            "vision_features":vision_features,
            "boxes":boxes,
            "task":"IR"
    }
    item[key_embedding] = method(batch).cpu().numpy()
    return item


def map_embed_passage(item, key_boxes: str, key_vision_features: str, key_text: str, key_embedding: str, method, kb, tokenizer, pool_strategy=None):
    kb_index = item['index']
    # get vision embedding from the kb
    vision_features = torch.Tensor(
        kb[kb_index][key_vision_features])
    vision_features = torch.squeeze(vision_features, dim=1)
    if key_boxes is None:
        boxes = torch.zeros(
               vision_features.shape[0],vision_features.shape[1], 4) 
    else:
        boxes = torch.Tensor(kb[kb_index][key_boxes])
        boxes = torch.squeeze(boxes, dim=1)
    input = tokenizer(
        item[key_text], return_tensors='pt', padding=True, truncation=True)
    batch = {
            "input_ids":input.input_ids,
            "attention_mask":input.attention_mask,
            "vision_features":vision_features,
            "boxes":boxes,
            "task":"IR"
    }
    item[key_embedding] = method(batch).cpu().numpy()
    return item

def dataset_embed_question(dataset_path: str,  config_question_path: str, config_passage_path: str, config_model_path: str, batch_size: int, **kwargs):
    dataset = load_from_disk(dataset_path)
    trainer = get_trainer(config_question_path, config_passage_path, config_model_path)
    tokenizer = trainer.tokenizer_question
    trainer.model.eval()
    kwargs.update(method=trainer.embedding_question, tokenizer=tokenizer)
    dataset = dataset.map(map_embed_question,
                          batched=True, batch_size=batch_size, fn_kwargs=kwargs)
    dataset.save_to_disk(dataset_path)


def dataset_embed_passage(dataset_path: str,  config_question_path: str, config_passage_path: str, config_model_path: str, kb_path: str, batch_size: int, **kwargs):
    dataset = load_from_disk(dataset_path)
    kb = load_from_disk(kb_path)
    trainer = get_trainer(config_question_path, config_passage_path, config_model_path)
    tokenizer = trainer.tokenizer_passage
    trainer.model.eval()
    kwargs.update(method=trainer.embedding_passage, kb=kb, tokenizer=tokenizer)
    dataset = dataset.map(map_embed_passage,
                          batched=True, batch_size=batch_size, fn_kwargs=kwargs)
    dataset.save_to_disk(dataset_path)

def get_trainer(config_question_path, config_passage_path, config_model_path):
    config_training = Config()
    config_training.local_rank = 0
    config_training.world_size = 0
    config_training.rank = 0
    config_training.distributed = False
    config_training.multiGPU = False
    if torch.cuda.is_available():
        torch.device("cuda", index=config_training.local_rank)
    else:
        torch.device("cuda", index=config_training.local_rank)
    config_encoder_question = Config.load_json(config_question_path)
    config_encoder_passage = Config.load_json(config_passage_path)
    config_model = Config.load_json(config_model_path)
    return Trainer(config_encoder_question, config_encoder_passage, config_model, config_training, train=False)

if __name__ == '__main__':
    # we consider that we have all we need to compute the embedding in the dataset
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, required=True)
    parser.add_argument('--type', type=str, required=True)
    parser.add_argument('--config_question_path', type=str, required=True)
    parser.add_argument('--config_passage_path', type=str, required=True)
    parser.add_argument('--config_model_path', type=str, required=True)
    parser.add_argument('--key_vision_features', type=str, required=True)
    parser.add_argument('--key_boxes', type=str, required=False, default=None)
    parser.add_argument('--key_embedding', type=str, required=True)
    parser.add_argument('--key_text', type=str, required=True)
    parser.add_argument('--kb_path', type=str, required=False)
    parser.add_argument('--batch_size', type=int, required=False, default=128)
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

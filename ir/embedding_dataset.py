"""embed dataset

Usage:

python -m embedding_dataset --dataset_path=<path> --type=<question or passage> --model_config_path=<path_to_config_model> \
    --key_boxes=<key_boxes> --key_vision_features --key_token=<key_token> --key_embedding=<key_embedding>
"""

from cvlep.trainer_base import Trainer
from datasets import load_from_disk, disable_caching
import argparse
import torch
from ir.utils import create_kwargs
from cvlep.utils import device
disable_caching()

# input
""" self,
        input_ids=None,
        attention_mask=None,

        vis_inputs=None,
        vis_attention_mask=None,

        inputs_embeds=None,
        head_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        return_pooled_output=False"""

# surement refactor quand on utilisera clip


def map_embed_question(item, key_boxes: str, key_vision_features: str, key_token: str, key_embedding: str, trainer: Trainer, **kwargs):
    vision_features=torch.Tensor(item[key_vision_features]).to(device)
    boxes = torch.Tensor(item[key_boxes]).to(device)
    inputs_ids=torch.LongTensor(item[key_token]).to(device)
    item[key_embedding] = trainer.embedding_question(
        vis_inputs=(vision_features,boxes),
        input_ids=inputs_ids,
        return_pooled_output=True
        ).pooler_output
    return item

# We could merge with function above but i think that we will need a separate function
# to deal with kb and passage
def map_embed_passage(item, key_boxes: str, key_vision_features: str, key_token: str, key_embedding: str, trainer: Trainer, **kwargs):
    param = dict()
    param.update(
        vis_inputs=(item[key_boxes], item[key_vision_features]),
        input_ids=torch.unsqueeze(item[key_token], dim = 0),
        return_pooled_output=True
    )
    item[key_embedding] = trainer.embedding_passage(**param)
    return item


def dataset_embed(type:str, dataset_path :str, model_config_path:str, key_boxes: str, key_vision_features: str, key_token,**kwargs):
    dataset = load_from_disk(dataset_path)
    trainer = Trainer(model_config_path)
    kwargs.update(
        trainer=trainer,
        key_token = key_token,
        key_boxes = key_boxes,
        key_vision_features=key_vision_features
        )
    if type == 'question':
        dataset = dataset.map(map_embed_question, batched=False, fn_kwargs=kwargs)
    elif type == 'passage':
        dataset = dataset.map(map_embed_passage, batched=False,fn_kwargs=kwargs)
    else:
        raise NotImplementedError()
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
    arg = parser.parse_args()
    kwargs = create_kwargs(arg)
    if arg.type not in ['question', 'passage']:
        raise NotImplementedError()
    else:
        dataset_embed(**kwargs)

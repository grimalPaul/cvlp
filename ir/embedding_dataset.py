"""embed dataset

Usage:

python -m embedding_dataset --path_dataset=<path> --type=<question or passage> --path_config_model=<path_to_config_model> \
    --key_boxes=<key_boxes> --key_vision_features --key_token=<key_token> --key_embedding=<key_embedding>
"""

from cvlep.trainer_base import Trainer
from datasets import load_from_disk, disable_caching
import argparse
import torch
from utils import create_kwargs
disable_caching()



# prend en entré ça
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
    param = dict()
    param.update(
        vis_inputs=(item[key_boxes], item[key_vision_features]),
        input_ids=torch.tensor(item[key_token]),
        return_pooled_output=True
    )
    item[key_embedding] = trainer.embedding_question(**param)
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


def dataset_embed(type:str, path_dataset :str, path_config_model:str, key_boxes: str, key_vision_features: str, key_token,**kwargs):
    dataset = load_from_disk(path_dataset)
    dataset.set_format(type='torch', columns=[key_token, key_boxes, key_vision_features])
    trainer = Trainer(path_config_model)
    kwargs.update(
        trainer=trainer,
        key_token = key_token,
        key_boxes = key_boxes,
        key_vision_features=key_vision_features
        )
    if type == 'question':
        dataset = dataset.map(map_embed_question, batched=True,
                              batch_size=64, fn_kwargs=kwargs)
    elif type == 'passage':
        dataset = dataset.map(map_embed_passage, batched=True,
                              batch_size=64, fn_kwargs=kwargs)
    else:
        raise NotImplementedError()
    dataset.save_to_disk(path_dataset)



if __name__ == '__main__':
    # we consider that we have all we need to compute the embedding in the dataset
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_dataset', type=str, required=True)
    parser.add_argument('--type', type=str, required=True)
    parser.add_argument('--path_config_model', type=str, required=True)
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

"""embed sentence with Sentence T5

Usage:
python -m processing.embedding_sentence_T5_only \
    --batch_size=64
    --model_path=path/to/folder_sentence_T5
    --key_text=
    --key_embedding=
    --dataset_path=path/to/dataset
"""
from processing.utils import create_kwargs
from sentence_transformers import SentenceTransformer
from datasets import load_from_disk, disable_caching
import argparse
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def map_sentence_T5(item,key_text,key_embedding,model):
    item[key_embedding]=model.encode((item[key_text]))
    return item

def sentence_T5(dataset_path,model_path, batch_size,**kwargs):
    disable_caching()
    model = SentenceTransformer(model_path)
    model.to(device)
    dataset = load_from_disk(dataset_path)
    kwargs.update(model=model)
    dataset=dataset.map(map_sentence_T5, fn_kwargs=kwargs, batch_size=batch_size)
    dataset.save_to_disk(dataset_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type =int, required=True, default = 64)
    parser.add_argument('--model_path', type=str, required=False)
    parser.add_argument('--dataset_path', type=str, required=True)
    parser.add_argument('--key_text', type=str, required=True)
    parser.add_argument('--key_embedding', type=str, required=True)

    kwargs = parser.parse_args()
    kwargs = create_kwargs(kwargs)
    sentence_T5(**kwargs)
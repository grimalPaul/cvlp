import argparse
from os import device_encoding
from socket import inet_aton
from turtle import title
from datasets import load_from_disk, disable_caching
import sys
import numpy as np
import torch
from tqdm import tqdm
import json
sys.path.append('/home/pgrimal/VL-T5/VL-T5/src')
sys.path.append('/home/pgrimal/VL-T5/VL-T5/')

from tokenization import VLT5TokenizerFast
from vqa import Trainer
from param import parse_args
from inference.modeling_frcnn import GeneralizedRCNN
from inference.processing_image import Preprocess
from inference.utils import Config

import json_stream

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# stream data
f = open('/scratch_global/stage_pgrimal/data/miniViQuAE/data/wikidata_id/vlt5/index2features.json', 'r')
data = json_stream.load(f)

def get_features(index):
    global data
    result = list()
    embed = data[index]
    for i in embed:
        vec1 = list()
        for j in i:
            vec2 = list()
            for k in j:
                vec2.append(k)
            vec1.append(vec2)
    result.append()
    return result

# get the trainer to acces to the jointEncoder
def get_trainer(backbone_path, load_path):
    args = parse_args(
        parse=False,
        backbone=backbone_path,
        load=load_path
    )
    args.gpu = 0
    trainer = Trainer(
        args,
        train=False
    )
    return trainer

# load Fastercnn
def load_frcnn(config_path, model_path):
    frcnn_cfg = Config.from_pretrained(config_path)
    frcnn = GeneralizedRCNN.from_pretrained(model_path, config=frcnn_cfg) 
    return frcnn, frcnn_cfg

# load preprocessor image
def load_img_preprocessor(frcnn_cfg):
    return Preprocess(frcnn_cfg) 

# tokenize
def add_tokenize(item, tokenizer, key):
    item['vlt5_input_id'] = tokenizer(item[key], return_tensors='pt', padding=True).input_ids
    return item

def tokenize_dataset(tokenizer, dataset_path, key):
    # on tokenize le text dans le dataset
    dataset = load_from_disk(dataset_path)
    fn_kwargs = {}
    fn_kwargs.update(tokenizer = tokenizer, key=key)
    dataset = dataset.map(add_tokenize, fn_kwargs=fn_kwargs)
    dataset.save_to_disk(dataset_path)

def item_visual_embedding(item, image_process, path_image, frcnn, frcnn_cfg):
    # a verifier pour le single_image = False
    # et pour le chemin d'acces
    print(f'{path_image}{item["image"]}')
    images, sizes, scales_yx = image_process(f'{path_image}{item["image"]}')
    #images, sizes, scales_yx = image_process(np.char.add(path_image,item["image"]).tolist(), single_image = False)
    output_dict = frcnn(
        images, 
        sizes, 
        scales_yx = scales_yx, 
        padding = 'max_detections', 
        max_detections = frcnn_cfg.max_detections, 
        return_tensors = 'pt' 
    )
    item['vlt5_normalized_boxes'] = output_dict.get("normalized_boxes")
    item['vlt5_features'] = output_dict.get("roi_features") 
    return item

def visual_embedding_dataset(dataset_path, config_path, model_path, path_image, batch_size):
    # on fait eùbedding des images dans le dataset
    dataset = load_from_disk(dataset_path)
    visual_model, cfg = load_frcnn(config_path, model_path)
    img_process = load_img_preprocessor(cfg)
    fn_kwargs = dict()
    fn_kwargs.update(image_process=img_process, path_image=path_image, frcnn = visual_model, frcnn_cfg = cfg)
    #dataset = dataset.map(item_visual_embedding, batched = True, batch_size = batch_size, fn_kwargs=fn_kwargs)
    #dataset = dataset.map(item_visual_embedding, num_proc = 8, fn_kwargs=fn_kwargs)
    dataset = dataset.map(item_visual_embedding, fn_kwargs=fn_kwargs)
    dataset.save_to_disk(dataset_path)

def embed(item,trainer):
    batch=dict()
    batch['input_ids'] = item['vlt5_input_id']
    batch['vis_feats'] = item['vlt5_features']
    batch['boxes'] = item['vlt5_normalized_boxes']
    item['vlt5_txt&img'] = trainer.model.encoder_step(batch)
    return item

def dataset_embed(dataset_path, backbone_path, epoch_path):
    dataset = load_from_disk(dataset_path)
    backbone_path=backbone_path
    load_path=epoch_path
    trainer = get_trainer(backbone_path, load_path)
    fn_kwargs = dict()
    fn_kwargs.update(trainer = trainer)
    dataset = dataset.map(embed, fn_kwargs=fn_kwargs)
    dataset.save_to_disk(dataset_path)


def map_vFeatures2images(item, index2boxes, index2features):
    # item['vlt5_features'] = index2features[str(item['index'])]
    item['vlt5_normalized_boxes'] = index2boxes[str(item['index'])]
    return item


def get_indx2features(kb, path, saved = False):
    # permet de charger les json avec index 2 embedding des images
    # mais fichier trop lourd pour les features donc autre techniquaes voir plus bas
    if saved:
        file_boxes = 'index2boxes.json'
        file_features = 'index2features.json'
        with open(f'{path}/{file_boxes}', 'r') as f:
            index2boxes=json.load(f)
            # index2boxes = {}
        with open(f'{path}/{file_features}', 'r') as f:
            index2features = {}
            # index2features = json.load(f)
    else :
        print('start loading')
        index2boxes = dict()
        index2features = dict()
        for split in kb:
            for index in kb[split]:
                for i in index['passage_index']:
                    index2boxes[i] = index['vlt5_normalized_boxes']
                    #index2features[i] = index['vlt5_features']
        print("load")
        """
        file_boxes = 'index2boxes.json'
        file_features = 'index2features.json'
        print('save files')
        with open(f'{path}/{file_boxes}', 'w') as f:
            json.dump(index2boxes, f)
        with open(f'{path}/{file_features}', 'w') as f:
            json.dump(index2features, f)"""
    return index2boxes, index2features
    

def map_kb2passage(kb_path, passages_path):
    # we want to add visual embedding to passage
    # out of memory donc simplify
    kb = load_from_disk(kb_path)
    passages = load_from_disk(passages_path)
    index2boxes, index2features = get_indx2features(kb, path='/scratch_global/stage_pgrimal/data/miniViQuAE/data/wikidata_id/vlt5', saved=False)
    fn_kwargs = {}
    fn_kwargs.update(index2boxes = index2boxes, index2features=index2features)
    passages = passages.map(map_vFeatures2images, fn_kwargs=fn_kwargs)
    passages.save_to_disk(passages_path)

def get_features(kb_path, path):
    # 2 dicts
    # dict index to title
    # dict title 2 features
    kb = load_from_disk(kb_path)
    title2features = dict()
    index2title = dict()
    for split in kb:
        for row in tqdm(kb[split]):
            for index in row['passage_index']:
                index2title[index] = row['wikipedia_title']
            title2features[row['wikipedia_title']] = row['vlt5_features']
    """path_index2title = 'index2title.json'
    path_title2features = 'title2features.json'
    with open(f'{path}/{path_index2title}', 'r') as f:
        index2title=json.load(f)
    with open(f'{path}/{path_title2features}', 'r') as f:
        title2features = json.load(f)
    print("files loaded")"""
    return index2title, title2features

def map_features_from_dict(item, index2title, title2features):
    index = int(item['index'])
    title = index2title[index]
    item['vlt5_features'] = title2features[title]
    return item

def map_passage_with_features(kb, passage_path):
    print('test passage')
    passages = load_from_disk(passage_path)
    index2title, title2features = get_features(kb, path='/scratch_global/stage_pgrimal/data/miniViQuAE/data/wikidata_id/vlt5')
    fn_kwargs = {}
    fn_kwargs.update(index2title = index2title, title2features = title2features)
    print('start mapping')
    passages  = passages.map(map_features_from_dict, fn_kwargs = fn_kwargs)
    passages.save_to_disk(passage_path)

if __name__ == '__main__':
    # cache useless because we load dataset from file
    disable_caching()
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='')
    parser.add_argument('--action', type=str, default='')
    parser.add_argument(
        '--path_image',
        type=str,
        default='/scratch_global/stage_pgrimal/data/miniViQuAE/data/dataset/miniCommons/')
    parser.add_argument('--key', type=str, default='input')
    
    # command to map visual embedding from kb to passage
    parser.add_argument('--kb_path', type=str, default='')
    parser.add_argument('--passages_path', type=str, default='')
    args = parser.parse_args()

    dataset_path = args.dataset
    if args.action == 'tokenize':
        # passage tokenize
        # load the tokenizer
        print('TOKENIZE')
        key = args.key
        tokenizer = VLT5TokenizerFast.from_pretrained('data/t5/t5-base')
        tokenize_dataset(tokenizer, dataset_path, key)
    elif args.action == "img":
        # Img embedding
        print("IMG Embedding")
        batch_size = 10
        path_image = args.path_image
        config_path = "data/frcnn_model/config.yaml"
        model_path = "data/frcnn_model/pytorch_model.bin"
        visual_embedding_dataset(dataset_path,config_path,model_path,path_image, batch_size)
    elif args.action == "encode":
        # Joint encoder
        print("Joint Encoder")
        backbone_path = 'data/t5_pretrained'
        epoch_path = 'VL-T5/snap/pretrain/VLT5/Epoch30'
        dataset_embed(dataset_path, backbone_path, epoch_path)
    elif args.action == "mapKB2PASSAGE":
        """print('map Visual embedding from kb to passage')
        kb_path = args.kb_path
        passages_path = args.passages_path 
        map_kb2passage(kb_path,passages_path)
        print('map features')"""
        kb_path = args.kb_path
        passages_path = args.passages_path 
        map_passage_with_features(kb_path, passages_path)
    else:
        print("Please precise the action")

"""
BROUILLON
---------

faire une copie de dataset v3

ajouter l'embedding des image dans le dataset et question tokenizé
utiliser la base de donnée qui ne différencie pas non humains, humains avec et sans visage, ajouter embedding du modèle

ajouter au dataset encodage question + image
ajouter à passage encodage des images
ajouter à passage encodage du passage q et de la question

faire un search et avoir les résultats
"""


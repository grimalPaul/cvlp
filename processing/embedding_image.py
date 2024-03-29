"""embed image with Faster RCNN or CLIP

Usage:
python -m processing.embedding_image \
    --type=CLIP/FasterRCNN \
    --batch_size=4
    # FasterRCNN
    --model_config_path=path/to/config.yaml
    --model=path/to/model.bin

    # CLIP
    --backbone=RN101

    # Dataset
    --dataset_path=path/to/dataset
    --key_image=key_image
    --key_image_embedding =key_image_embedding
    --image_path=path/to/image
    --log_path=path/to/save_error
"""

from pathlib import Path
from cvlep.VLT5.inference.modeling_frcnn import GeneralizedRCNN
from cvlep.VLT5.inference.processing_image import Preprocess
from cvlep.VLT5.inference.utils import Config
from datasets import load_from_disk, disable_caching
import argparse
from processing.utils import create_kwargs
from PIL import Image
import torch
from cvlep.CLIPT5.vis_encoder import get_vis_encoder, _transform
from cairosvg import svg2png
from io import BytesIO
import json

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

disable_caching()

log_file = []

# load Fastercnn
def load_frcnn(config_path, model_path):
    frcnn_cfg = Config.from_pretrained(config_path)
    frcnn = GeneralizedRCNN.from_pretrained(model_path, config=frcnn_cfg)
    return frcnn, frcnn_cfg

# load preprocessor image
def load_img_preprocessor(frcnn_cfg):
    return Preprocess(frcnn_cfg)

# CLIP
def open_image(path):
    if str(path)[-4:] == ".svg":
        png = svg2png(file_obj=open(path,'r'))
        return Image.open(BytesIO(png))
    else:
        return Image.open(path)

def item_CLIP_embedding(item, key_image, key_image_embedding, image_path, transform, vis_encoder, batch_size):
    # mapping function to embed images with clip
    # can embed single image or list of image
    global log_file
    list_images = item[key_image]

    with torch.no_grad():
        try:
            if isinstance(list_images, list):
                images = list()
                for image_name in list_images:
                    image = transform(open_image(Path(image_path)/image_name))
                    images.append(image)
                images = torch.stack(images)
                if images.shape[0] > batch_size:
                    nb_batch = images.shape[0] // batch_size
                    remainder= images.shape[0] % batch_size
                    features = torch.empty((images.shape[0],49,2048))
                    for i in range(0, nb_batch*batch_size, batch_size):
                        features[i:batch_size+i,:,:],_=vis_encoder((images[i:batch_size+i,:,:,:]).to(device))
                    if remainder > 0:
                        features[-remainder:,:,:],_ =vis_encoder(
                            (images[-remainder:,:,:,:]).to(device)
                        )
                    item[f'{key_image_embedding}_features'] = features.cpu().numpy()
                else:
                    features,_ =vis_encoder(
                            (images).to(device)
                        )
                    item[f'{key_image_embedding}_features'] = features.cpu().numpy()
            else:
                images = transform(open_image(Path(image_path)/list_images))
                images = torch.unsqueeze(images, dim=0)
                features,_ = vis_encoder((images).to(device))
                item[f"{key_image_embedding}_features"] = features.cpu().numpy()
        except:
            item[f'{key_image_embedding}_features'] = None
            log_file.append(item['wikipedia_title'])
    return item

def embed_with_CLIP(dataset_path, backbone, **kwargs):
    dataset = load_from_disk(dataset_path)
    model = get_vis_encoder(backbone=backbone, adapter_type=None, image_size=eval("(224,224)")[0])
    model = model.to(device)
    model.eval()
    kwargs.update(
        transform = _transform(eval("(224,224)")[0]),
        vis_encoder = model
    )
    dataset = dataset.map(item_CLIP_embedding, fn_kwargs=kwargs)
    dataset.save_to_disk(dataset_path)


def item_fasterRCNN_embedding(item, key_image, key_image_embedding, image_process, image_path, frcnn, frcnn_cfg, batch_size):
    list_images = item[key_image]
    global log_file
    with torch.no_grad():
        try:
            if isinstance(list_images,list):
                images = list()
                for image_name in list_images:
                    images.append(str(Path(image_path) / image_name))
                images, sizes, scales_yx = image_process(images)
                print(images.shape)
                if images.shape[0] > batch_size:
                    nb_batch = images.shape[0] // batch_size
                    remainder= images.shape[0] % batch_size
                    boxes = torch.empty((images.shape[0],frcnn_cfg.max_detections,4))
                    features = torch.empty((images.shape[0],frcnn_cfg.max_detections, 2048))
                    boxes = boxes.to(device)
                    features = features.to(device)
                    for i in range(0, nb_batch*batch_size, batch_size):
                        output_dict = frcnn(
                            images[i:batch_size+i,:,:,:],
                            sizes[i:batch_size+i,:],
                            scales_yx=scales_yx[i:batch_size+i,:],
                            padding='max_detections',
                            max_detections=frcnn_cfg.max_detections,
                            return_tensors='pt'
                        )
                        boxes[i:batch_size+i,:,:]=output_dict.get("normalized_boxes")
                        features[i:batch_size+i,:,:]=output_dict.get("roi_features")
                    if remainder > 0:
                        output_dict = frcnn(
                            images[-remainder:,:,:,:],
                            sizes[-remainder:,:],
                            scales_yx=scales_yx[-remainder:,:],
                            padding='max_detections',
                            max_detections=frcnn_cfg.max_detections,
                            return_tensors='pt'
                        )
                        boxes[-remainder:,:,:]=output_dict.get("normalized_boxes")
                        features[-remainder:,:,:]=output_dict.get("roi_features")
                    print(boxes.shape)
                    print(features.shape)
                    item[f'{key_image_embedding}_boxes'] = boxes.cpu().numpy()
                    item[f'{key_image_embedding}_features'] = features.cpu().numpy()
                else:
                    output_dict = frcnn(
                            images,
                            sizes,
                            scales_yx=scales_yx,
                            padding='max_detections',
                            max_detections=frcnn_cfg.max_detections,
                            return_tensors='pt'
                        )
                    item[f'{key_image_embedding}_boxes']=output_dict.get("normalized_boxes")
                    item[f'{key_image_embedding}_features']=output_dict.get("roi_features")
            else:
                images, sizes, scales_yx = image_process(str(Path(image_path)/list_images))
                output_dict = frcnn(
                    images,
                    sizes,
                    scales_yx=scales_yx,
                    padding='max_detections',
                    max_detections=frcnn_cfg.max_detections,
                    return_tensors='pt'
                )
                item[f'{key_image_embedding}_boxes'] = output_dict.get("normalized_boxes").cpu().numpy()
                item[f'{key_image_embedding}_features'] = output_dict.get("roi_features").cpu().numpy()
        except:
            item[f'{key_image_embedding}_boxes'] = None
            item[f'{key_image_embedding}_features'] = None
            log_file.append(item['wikipedia_title'])
    return item


def embed_with_fasterRCNN(dataset_path, model, model_config_path, **kwargs):
    dataset = load_from_disk(dataset_path)
    visual_model, cfg = load_frcnn(model_config_path, model)
    img_process = load_img_preprocessor(cfg)
    visual_model = visual_model.to(device)
    kwargs.update(
        image_process=img_process,
        frcnn=visual_model,
        frcnn_cfg=cfg
    )
    dataset = dataset.map(item_fasterRCNN_embedding, fn_kwargs=kwargs)
    dataset.save_to_disk(dataset_path)


def embed_with_fasterRCNN_and_CLIP(dataset_path, model, model_config_path, **kwargs):
    pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', type=str, required=True)
    parser.add_argument('--batch_size', type =int, required=True, default = 4)
    # FasterRCNN
    parser.add_argument('--model_config_path', type=str, required=False)
    parser.add_argument('--model', type=str, required=False)

    # CLIP
    parser.add_argument('--backbone', type=str, required=False)

    # Dataset
    parser.add_argument('--dataset_path', type=str, required=True)
    parser.add_argument('--key_image', type=str, required=True)
    parser.add_argument('--key_image_embedding', type=str, required=True)
    parser.add_argument('--image_path', type=str, required=False)
    parser.add_argument('--log_path', type=str, required=False, default='')
    arg = parser.parse_args()
    kwargs = create_kwargs(arg)
    kwargs.pop('type')
    log_path = kwargs.pop('log_path')
    if arg.type == 'CLIP':
        kwargs.pop("model_config_path")
        kwargs.pop("model")
        embed_with_CLIP(**kwargs)
    elif arg.type == 'FasterRCNN':
        kwargs.pop("backbone")
        embed_with_fasterRCNN(**kwargs)
    else:
        raise NotImplementedError()
    if log_path != '':
        print(log_file)
        print(len(log_file))
        with open(log_path, 'w') as f:
            json.dump({"error":log_file},f)
    

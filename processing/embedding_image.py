"""embed image

Usage:
python -m ir.embedding_image --type=encode_image --dataset_path=<path> --model_config_path=<path_to_config_model> \
    --key_image=<key_text> --key_image_embedding=<key_token> --image_path=<image_path> --model=<pytorch_model.bin>
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

disable_caching()

# load Fastercnn
def load_frcnn(config_path, model_path):
    frcnn_cfg = Config.from_pretrained(config_path)
    frcnn = GeneralizedRCNN.from_pretrained(model_path, config=frcnn_cfg)
    return frcnn, frcnn_cfg

# load preprocessor image
def load_img_preprocessor(frcnn_cfg):
    return Preprocess(frcnn_cfg)

# CLIP
# model = get_vis_encoder(backbone='RN50', adapter_type=None, image_size=eval("(224,224)")[0])
# model.eval() # je pense pour ne pas avoir de gradient
def open_image(path):
    if path[-4:] == ".svg":
        png = svg2png(file_obj=open(path,'r'))
        return Image.open(BytesIO(png))
    else:
        return Image.open(path)
def item_CLIP_embedding(item, key_image, key_image_embedding, image_path, transform, vis_encoder, batch_size):
    # mapping function to embed images with clip
    # can embed single image or list of image
    list_images = item[key_image]
    if isinstance(list_images, list):
        images = list()
        for image_name in list_images:
            image = transform(open_image(Path(image_path)/image_name))
            images.append(image)
        images = torch.stack(images)
    else:
        images = transform(open_image(f'{image_path}{list_images}'))
    #TODO : fake batch size for single images ?
    #TODO : witch output ?
    output = _,_ = vis_encoder(images)
    item[f"{key_image_embedding}_features"] = output
    return item

def embed_with_CLIP(dataset_path, backbone, **kwargs):
    dataset = load_from_disk(dataset_path)
    model = get_vis_encoder(backbone=backbone, adapter_type=None, image_size=eval("(224,224)")[0])
    kwargs.update(
        transform = _transform,
        vis_encoder = model
    )
    dataset = dataset.map(item_CLIP_embedding, fn_kwargs=kwargs)
    dataset.save_to_disk(dataset_path)


def item_fasterRCNN_embedding(item, key_image, key_image_embedding, image_process, image_path, frcnn, frcnn_cfg, batch_size):
    list_images = item[key_image]
    #TODO: batch max number of image and concatenate results !
    if isinstance(list_images,list):
        images = list()
        for image_name in list_images:
            images.append(str(Path(image_path) / image_name))
        # if len(images) > batch_size:
        #     nb_batch = images.shape[0] // batch_size
        #     remainder= images.shape[0] % batch_size
        #     boxes = []
        #     features = []
        #     for i in range(0, nb_batch*batch_size, batch_size):
        #         im = images[:batch_size]
        #         images = images[:batch_size]
        #         images, sizes, scales_yx = image_process(im)
            
        images, sizes, scales_yx = image_process(images)
        if images.shape[0] > batch_size:
            nb_batch = images.shape[0] // batch_size
            remainder= images.shape[0] % batch_size
            boxes = torch.empty((images.shape[0],frcnn_cfg.max_detections,4))
            features = torch.empty((images.shape[0],frcnn_cfg.max_detections, 2048))
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
            item[f'{key_image_embedding}_boxes'] = boxes
            item[f'{key_image_embedding}_features'] = features
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
        item[f'{key_image_embedding}_boxes'] = output_dict.get("normalized_boxes")
        item[f'{key_image_embedding}_features'] = output_dict.get("roi_features")
    return item


def embed_with_fasterRCNN(dataset_path, model, model_config_path, **kwargs):
    dataset = load_from_disk(dataset_path)
    visual_model, cfg = load_frcnn(model_config_path, model)
    img_process = load_img_preprocessor(cfg)
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
    arg = parser.parse_args()
    kwargs = create_kwargs(arg)
    kwargs.pop('type')
    if arg.type == 'CLIP':
        kwargs.pop("model_config_path")
        kwargs.pop("model")
        embed_with_CLIP(**kwargs)
    elif arg.type == 'FasterRCNN':
        kwargs.pop("backbone")
        embed_with_fasterRCNN(**kwargs)
    else:
        raise NotImplementedError()

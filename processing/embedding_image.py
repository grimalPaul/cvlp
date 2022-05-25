"""embed image

Usage:
python -m ir.embedding_image --type=encode_image --dataset_path=<path> --model_config_path=<path_to_config_model> \
    --key_image=<key_text> --key_image_embedding=<key_token> --image_path=<image_path> --model=<pytorch_model.bin>
"""

from cvlep.VLT5.inference.modeling_frcnn import GeneralizedRCNN
from cvlep.VLT5.inference.processing_image import Preprocess
from cvlep.VLT5.inference.utils import Config
from datasets import load_from_disk, disable_caching
import argparse
from processing.utils import create_kwargs

disable_caching()

# load Fastercnn
def load_frcnn(config_path, model_path):
    frcnn_cfg = Config.from_pretrained(config_path)
    frcnn = GeneralizedRCNN.from_pretrained(model_path, config=frcnn_cfg)
    return frcnn, frcnn_cfg

# load preprocessor image
def load_img_preprocessor(frcnn_cfg):
    return Preprocess(frcnn_cfg)


def item_visual_embedding(item, key_image, key_image_embedding, image_process, image_path, frcnn, frcnn_cfg, **kwargs):
    images, sizes, scales_yx = image_process(f'{image_path}{item[key_image]}')
    output_dict = frcnn(
        images,
        sizes,
        scales_yx=scales_yx,
        padding='max_detections',
        max_detections=frcnn_cfg.max_detections,
        return_tensors='pt'
    )
    item[f'{key_image_embedding}_normalized_boxes'] = output_dict.get(
        "normalized_boxes")
    item[f'{key_image_embedding}_features'] = output_dict.get("roi_features")
    return item


def embed_image_dataset(dataset_path, model, model_config_path, **kwargs):
    dataset = load_from_disk(dataset_path)
    visual_model, cfg = load_frcnn(model_config_path, model)
    img_process = load_img_preprocessor(cfg)
    kwargs.update(
        image_process=img_process,
        frcnn=visual_model,
        frcnn_cfg=cfg
    )
    dataset = dataset.map(item_visual_embedding, fn_kwargs=kwargs)
    dataset.save_to_disk(dataset_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, required=True)
    parser.add_argument('--type', type=str, required=True)
    parser.add_argument('--model_config_path', type=str, required=True)

    parser.add_argument('--model', type=str, required=False)
    parser.add_argument('--key_image', type=str, required=False)
    parser.add_argument('--key_image_embedding', type=str, required=False)
    parser.add_argument('--image_path', type=str, required=False)
    arg = parser.parse_args()
    kwargs = create_kwargs(arg)
    kwargs.pop('type')
    if arg.type == 'embedding_image':
        embed_image_dataset(**kwargs)
    else:
        raise NotImplementedError()
